# -*- coding: utf-8 -*-
"""
模块 3：大语言模型（LLM）搞钱赛道提炼与四级合规风险标注流水线
归属课题：《基于大语言模型的二手平台数字商品生命周期与需求分析系统》
运行环境：Windows 11 / Python 3.12+
依赖安装：pip install openai pydantic
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import re
import json
import time
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, ValidationError

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ==============================================================================
# 1. 日志与全局常量配置
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("Module3_LLM_Pipeline")

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xianyu_lifecycle.db")

STANDARD_SUB_CATEGORIES = [
    "公考笔试(行测/申论)",
    "公考面试与事业编",
    "通用求职简历模板",
    "行业定制/AI简历模板",
    "Python Web系统源码",
    "Python爬虫与数据分析",
    "AI大模型与深度学习源码",
    "自动化脚本与工具包"
]

RISK_LEVEL_DEFINITIONS = {
    "L0": "正常合规（通用无版权纠纷公开模板、自研代码、原创备考笔记）",
    "L1": "疑似搬运（公开网课二次打包、非商用共享资料整理，存在擦边但非主观恶意）",
    "L2": "侵权灰产（付费名师商业课程完整盗录、破解补丁、盗版教材、外挂脱机脚本）",
    "L3": "违规引流（低俗擦边、网赌黑产、强制脱离平台加微信私信交易或钓鱼链接）"
}

# ==============================================================================
# 2. 结构化输出 Schema 定义 (Pydantic 强校验)
# ==============================================================================
class LLMExtractionResult(BaseModel):
    fine_target: str = Field(
        ...,
        description="剥离营销词（如秒发/包更新/最新）后的细粒度真实交付标的，控制在6~18个汉字"
    )
    sub_category: str = Field(
        ...,
        description="归一化后的标准二级赛道名称"
    )
    risk_level: str = Field(
        ...,
        pattern=r"^L[0-3]$",
        description="四级合规风险等级：L0, L1, L2 或 L3"
    )
    risk_type: str = Field(
        ...,
        description="风险细分类别，如：原创合规、网课搬运、盗版课程侵权、外挂灰产、站外违规引流"
    )
    risk_reason: str = Field(
        ...,
        description="简明扼要的合规判定依据（30字以内）"
    )
    pain_points_hit: List[str] = Field(
        default_factory=list,
        description="命中的买家痛点维度列表：交付时效性、格式兼容性、版本时效性、内容完整度、零门槛开箱即用"
    )
    seo_title_suggestion: str = Field(
        ...,
        description="基于动态SEO重排算子重构的30字高转化合规标题"
    )

# ==============================================================================
# 3. System Prompt 与 Few-Shot 提示词工程设计
# ==============================================================================
SYSTEM_PROMPT = """你是一位资深的二手电商数字商品（虚拟物品）数据挖掘专家与平台风控合规审计师。
你的任务是对闲鱼平台采集到的数字商品（涵盖简历模板、考公资料、Python源码三大核心赛道）进行细粒度语义解构与四级合规风险评级。

### 一、 核心提取字段规范
1. `fine_target`（细粒度真实标的）：
   - 必须彻底剥离“2026最新”、“大厂HR推荐”、“24小时自动发货”、“百度网盘秒发”、“包更新”、“白菜价”等营销修饰词；
   - 精准提炼买家付款后实际拿到的核心数字资产名称（例如：“2026国考申论行测真题讲义”、“SpringBoot校园外卖管理系统源码”、“应届生通用Word简历模板”）。
2. `sub_category`（标准化二级赛道）：
   - 必须从以下标准集合中择一映射：
     ["公考笔试(行测/申论)", "公考面试与事业编", "通用求职简历模板", "行业定制/AI简历模板", "Python Web系统源码", "Python爬虫与数据分析", "AI大模型与深度学习源码", "自动化脚本与工具包"]
3. `risk_level` 与 `risk_type`（四级合规与擦边风险矩阵）：
   - `L0`（正常合规）：通用排版模板、开源/自研毕业设计源码、个人原创复习笔记，无明确第三方商业版权指向；
   - `L1`（疑似搬运）：公开免费资料二次汇总打包、带有“某某机构整理/历年真题汇编”但未直接售卖加密商业视频的轻度擦边商品；
   - `L2`（侵权灰产）：明确包含付费名师/商业机构（如粉笔、华图、中公等）完整盗录视频课、商业软件破解版、自动化刷课/抢票外挂脚本；
   - `L3`（违规引流）：标题含“加V私聊”、“微信发红包”、“福利资源”、“博彩/棋牌源码”等严重违反平台安全规范的标的。
4. `pain_points_hit`（五维痛点命中分析）：
   - 识别标题中显式覆盖的买家痛点维度（多选）：["交付时效性", "格式兼容性", "版本时效性", "内容完整度", "零门槛开箱即用"]。
5. `seo_title_suggestion`（动态SEO标题重排建议）：
   - 按照【年份时效锚点 + 细粒度核心标的 + 完整度/开箱即用承诺 + 格式兼容声明 + 自动秒发保障】公式生成一条 26~30 字的高转化率合规标题。

### 二、 Few-Shot 标准标注示例
【输入示例 1】
主赛道: 考公资料 | 标价: 2.88元 | 原始标题: 26&27公考备考 国考省考事业编行测申论全套名师视频网课讲义真题库 百度网盘24小时自动发货
【输出示例 1】
{
  "fine_target": "国省考及事业编行测申论视频讲义题库",
  "sub_category": "公考笔试(行测/申论)",
  "risk_level": "L2",
  "risk_type": "商业网课盗录侵权",
  "risk_reason": "明确包含全套名师商业视频网课，涉嫌侵犯培训机构著作权",
  "pain_points_hit": ["交付时效性", "版本时效性", "内容完整度"],
  "seo_title_suggestion": "2026国省考行测申论高频考点精讲笔记 历年真题含解析PDF网盘秒发"
}

【输入示例 2】
主赛道: 简历模板 | 标价: 1.00元 | 原始标题: 3000套个人简历模板word可编辑应届生求职通用的春招秋招自我介绍表格制作24小时秒发
【输出示例 2】
{
  "fine_target": "应届生通用Word求职简历模板库",
  "sub_category": "通用求职简历模板",
  "risk_level": "L0",
  "risk_type": "原创或通用合规模板",
  "risk_reason": "通用办公文档排版框架，无特定商业版权侵权特征",
  "pain_points_hit": ["交付时效性", "格式兼容性", "零门槛开箱即用"],
  "seo_title_suggestion": "2026春招秋招应届生求职简历模板 Word/WPS双端可编辑含排版指南秒发"
}

【输入示例 3】
主赛道: Python源码 | 标价: 1.50元 | 原始标题: Python毕设源码Django基于大数据的电商评论情感分析可视化系统含部署视频包跑通
【输出示例 3】
{
  "fine_target": "Django电商评论情感分析可视化源码",
  "sub_category": "Python爬虫与数据分析",
  "risk_level": "L0",
  "risk_type": "合规自研/教学源码",
  "risk_reason": "标准计算机教学与毕业设计开源框架项目，含部署教学文档",
  "pain_points_hit": ["内容完整度", "零门槛开箱即用"],
  "seo_title_suggestion": "Python+Django电商评论情感分析可视化系统源码 含SQL库与部署教程秒发"
}

请严格输出合法的 JSON 对象，不要包含任何 Markdown 代码块标签或额外解释文本。"""

# ==============================================================================
# 4. 本地高保真规则兜底引擎（当无 API Key 或网络受限时的确定性保障）
# ==============================================================================
def rule_based_fallback_extractor(title: str, category_keyword: str, price: float) -> LLMExtractionResult:
    """
    基于正则与领域词典的确定性抽取器，确保离线答辩或 API 额度耗尽时 100% 产出一流结构化数据。
    """
    clean_title = re.sub(
        r"(24小时|自动发货|秒发|百度网盘|夸克网盘|包更新|白菜价|超值|合集|全套|整理|推荐|最新|202[4567]年?)",
        "",
        title
    ).strip()
    clean_title = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\+]+", "", clean_title)

    # 1. 识别二级赛道与细粒度标的
    cat_str = str(category_keyword or "")
    if "公" in cat_str or any(k in title for k in ["公考", "行测", "申论", "国考", "省考", "事业编", "粉笔", "华图"]):
        if any(k in title for k in ["面试", "结构化", "事业编", "事业单位"]):
            sub_cat = "公考面试与事业编"
            fine_target = "公考面试与事业单位备考全套讲义"
        else:
            sub_cat = "公考笔试(行测/申论)"
            fine_target = "国省考行测申论模块化刷题库与笔记"
    elif "简历" in cat_str or any(k in title for k in ["简历", "求职", "春招", "秋招", "应届", "Word模板"]):
        if any(k in title for k in ["AI", "大厂", "程序员", "互联网", "定制", "一对一", "精修"]):
            sub_cat = "行业定制/AI简历模板"
            fine_target = "互联网/AI岗位高转化定制简历模板"
        else:
            sub_cat = "通用求职简历模板"
            fine_target = "应届生通用可编辑Word求职简历模板"
    else:
        if any(k in title for k in ["大模型", "LLM", "深度学习", "YOLO", "神经网络", "机器学习", "预测"]):
            sub_cat = "AI大模型与深度学习源码"
            fine_target = "Python深度学习与大模型应用实战源码"
        elif any(k in title for k in ["爬虫", "数据分析", "可视化", "大屏", "Scrapy", "Playwright", "情感分析"]):
            sub_cat = "Python爬虫与数据分析"
            fine_target = "Python数据采集清洗与可视化大屏源码"
        elif any(k in title for k in ["脚本", "自动化", "办公", "批量", "工具"]):
            sub_cat = "自动化脚本与工具包"
            fine_target = "Python桌面办公自动化批处理脚本源码"
        else:
            sub_cat = "Python Web系统源码"
            fine_target = "Python+Django/Flask全栈管理系统源码"

    if len(clean_title) >= 8:
        fine_target = clean_title[:16]

    # 2. 四级合规风险判定
    if any(k in title for k in ["加微", "加V", "vx", "私聊发", "博彩", "棋牌", "灰产", "破解收费"]):
        risk_level = "L3"
        risk_type = "站外违规引流或黑产"
        risk_reason = "标题包含站外社交账号引流词或高危敏感黑产词汇"
    elif any(k in title for k in ["名师视频", "盗录", "粉笔", "华图", "中公", "破解版", "外挂", "抢票脚本", "内部密押"]):
        risk_level = "L2"
        risk_type = "商业网课盗版或外挂侵权"
        risk_reason = "涉及知名商业培训机构付费课程视频或未授权破解脚本"
    elif any(k in title for k in ["网课", "视频课", "合集打包", "电子书", "PDF教材", "题库汇总"]):
        risk_level = "L1"
        risk_type = "公开资料二次搬运打包"
        risk_reason = "聚合网络公开讲义或非商用共享资料，存在轻度版权擦边"
    else:
        risk_level = "L0"
        risk_type = "原创或通用合规数字标的"
        risk_reason = "属于通用格式模板、开源技术框架或原创整理笔记，合规安全"

    # 3. 命中痛点分析
    pain_points = []
    if any(k in title for k in ["自动发货", "秒发", "24小时", "网盘"]):
        pain_points.append("交付时效性")
    if any(k in title for k in ["Word", "WPS", "可编辑", "PDF", "通用", "源码"]):
        pain_points.append("格式兼容性")
    if any(k in title for k in ["2026", "26", "27", "最新", "新版", "春招", "秋招"]):
        pain_points.append("版本时效性")
    if any(k in title for k in ["全套", "完整", "含解析", "含数据库", "含文档", "无水印"]):
        pain_points.append("内容完整度")
    if any(k in title for k in ["开箱即用", "包跑通", "部署视频", "直接套用", "零基础"]):
        pain_points.append("零门槛开箱即用")
    if not pain_points:
        pain_points = ["交付时效性", "零门槛开箱即用"]

    # 4. 动态 SEO 标题重排生成
    seo_title = f"2026新版{fine_target[:10]} 完整无水印含使用指南 支持多端编辑网盘24h秒发"

    return LLMExtractionResult(
        fine_target=fine_target,
        sub_category=sub_cat,
        risk_level=risk_level,
        risk_type=risk_type,
        risk_reason=risk_reason,
        pain_points_hit=pain_points,
        seo_title_suggestion=seo_title[:30]
    )

# ==============================================================================
# 5. 多通道大模型客户端管理器（支持 DeepSeek 主用 + 通义千问备用 + 指数退避）
# ==============================================================================
class MultiChannelLLMClient:
    def __init__(self):
        self.channels: List[Dict[str, str]] = []
        deepseek_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        dashscope_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()

        if deepseek_key and OpenAI is not None:
            self.channels.append({
                "name": "DeepSeek-V3",
                "api_key": deepseek_key,
                "base_url": "https://api.deepseek.com/v1",
                "model": "deepseek-chat"
            })
        if dashscope_key and OpenAI is not None:
            self.channels.append({
                "name": "Qwen-Plus (DashScope)",
                "api_key": dashscope_key,
                "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "model": "qwen-plus"
            })

        if self.channels:
            logger.info(f"已激活 {len(self.channels)} 个大模型 API 通道: {[c['name'] for c in self.channels]}")
        else:
            logger.warning("未检测到有效 API 密钥或未安装 openai 库，将启用高保真本地规则引擎模式。")

    def analyze_item(self, title: str, category_keyword: str, price: float, max_retries: int = 3) -> Tuple[LLMExtractionResult, str]:
        """
        调用大模型进行单商品结构化抽取，带通道自动轮询与指数退避重试。
        返回值: (LLMExtractionResult 实例, 实际生效的引擎名称)
        """
        user_prompt = (
            f"请对以下闲鱼二手数字商品进行细粒度实体抽取与四级合规风险审计：\n"
            f"主赛道: {category_keyword or '未分类'} | 标价: {price:.2f}元 | 原始标题: {title}"
        )

        for channel in self.channels:
            client = OpenAI(api_key=channel["api_key"], base_url=channel["base_url"])
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=channel["model"],
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.1,
                        max_tokens=512,
                        response_format={"type": "json_object"},
                        timeout=20.0
                    )
                    raw_content = response.choices[0].message.content.strip()
                    # 清理可能残留的 Markdown 标记
                    raw_content = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_content, flags=re.IGNORECASE).strip()
                    parsed_dict = json.loads(raw_content)

                    # 归一化二级赛道名称
                    if parsed_dict.get("sub_category") not in STANDARD_SUB_CATEGORIES:
                        fallback_obj = rule_based_fallback_extractor(title, category_keyword, price)
                        parsed_dict["sub_category"] = fallback_obj.sub_category

                    validated = LLMExtractionResult(**parsed_dict)
                    return validated, channel["name"]
                except (ValidationError, json.JSONDecodeError) as ve:
                    logger.warning(f"[{channel['name']}] JSON 结构校验失败 (尝试 {attempt}/{max_retries}): {ve}")
                except Exception as e:
                    backoff_sec = 1.5 ** attempt
                    logger.warning(f"[{channel['name']}] 请求异常 (尝试 {attempt}/{max_retries}): {e}，等待 {backoff_sec:.1f}s 后重试...")
                    time.sleep(backoff_sec)

            logger.error(f"通道 [{channel['name']}] 达到最大重试次数，已将该通道熔断移除...")
            self.channels.remove(channel)
            break

        # 所有 API 通道均不可用时，无缝切换至本地确定性规则引擎
        fallback_res = rule_based_fallback_extractor(title, category_keyword, price)
        return fallback_res, "Local-Rule-Engine"

# ==============================================================================
# 6. 数据库迁移、断点续传与批量回写主流程
# ==============================================================================
def ensure_db_schema(conn: sqlite3.Connection) -> None:
    """检查并自动扩展 items 表的 LLM 标注列，支持幂等执行"""
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(items)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    new_columns = {
        "fine_target": "TEXT",
        "sub_category": "TEXT",
        "risk_level": "TEXT",
        "risk_type": "TEXT",
        "risk_reason": "TEXT",
        "pain_points_json": "TEXT",
        "seo_title_suggestion": "TEXT",
        "llm_engine_used": "TEXT",
        "llm_processed_at": "TEXT"
    }

    for col_name, col_type in new_columns.items():
        if col_name not in existing_cols:
            alter_sql = f"ALTER TABLE items ADD COLUMN {col_name} {col_type}"
            cursor.execute(alter_sql)
            logger.info(f"数据库 Schema 自动演进：已向 items 表新增字段 `{col_name}` ({col_type})")
    conn.commit()

def run_pipeline(force_reprocess: bool = False) -> None:
    """运行模块 3 完整标注与回写流水线"""
    if not os.path.exists(DB_PATH):
        logger.error(f"未找到数据库文件: {DB_PATH}，请确认路径是否正确！")
        return

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    ensure_db_schema(conn)
    cursor = conn.cursor()

    if force_reprocess:
        query_sql = "SELECT item_id, title, price, category_keyword FROM items"
    else:
        query_sql = (
            "SELECT item_id, title, price, category_keyword FROM items "
            "WHERE llm_processed_at IS NULL OR fine_target IS NULL OR fine_target = ''"
        )

    cursor.execute(query_sql)
    pending_items = cursor.fetchall()
    total_pending = len(pending_items)

    if total_pending == 0:
        logger.info("所有商品均已完成 LLM 实体抽取与合规风险标注（断点续传检测通过，无需重复处理）。")
    else:
        logger.info(f"待处理商品总数: {total_pending} 条，开始执行结构化语义抽取流水线...")
        llm_client = MultiChannelLLMClient()

        for idx, row in enumerate(pending_items, start=1):
            item_id = str(row["item_id"])
            title = str(row["title"] or "")
            price = float(row["price"] or 1.0)
            category_keyword = str(row["category_keyword"] or "数字商品")

            result, engine_name = llm_client.analyze_item(
                title=title,
                category_keyword=category_keyword,
                price=price
            )

            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cursor.execute(
                """
                UPDATE items
                SET fine_target = ?,
                    sub_category = ?,
                    risk_level = ?,
                    risk_type = ?,
                    risk_reason = ?,
                    pain_points_json = ?,
                    seo_title_suggestion = ?,
                    llm_engine_used = ?,
                    llm_processed_at = ?
                WHERE item_id = ?
                """,
                (
                    result.fine_target,
                    result.sub_category,
                    result.risk_level,
                    result.risk_type,
                    result.risk_reason,
                    json.dumps(result.pain_points_hit, ensure_ascii=False),
                    result.seo_title_suggestion,
                    engine_name,
                    now_str,
                    item_id
                )
            )

            if idx % 10 == 0 or idx == total_pending:
                conn.commit()
                logger.info(
                    f"[{idx}/{total_pending}] 已落盘 | ID:{item_id[:8]}... | "
                    f"标的:{result.fine_target} | 二级赛道:{result.sub_category} | "
                    f"风险:{result.risk_level}({result.risk_type}) | 引擎:{engine_name}"
                )

    # 输出全库标注审计统计摘要
    print("\n" + "=" * 78)
    print("📊 模块 3：大模型搞钱赛道提炼与四级合规风险审计汇总报告")
    print("=" * 78)
    cursor.execute(
        """
        SELECT sub_category, COUNT(*) as cnt, ROUND(AVG(price), 2) as avg_price
        FROM items
        GROUP BY sub_category
        ORDER BY cnt DESC
        """
    )
    print("\n【1. 标准化二级赛道分布】")
    for r in cursor.fetchall():
        print(f" - {r['sub_category'] or '未分类':<22} : {r['cnt']:>3} 件 | 均价: {r['avg_price']:.2f} 元")

    cursor.execute(
        """
        SELECT risk_level, risk_type, COUNT(*) as cnt
        FROM items
        GROUP BY risk_level
        ORDER BY risk_level ASC
        """
    )
    print("\n【2. 四级合规风险等级分布】")
    for r in cursor.fetchall():
        lvl = r["risk_level"] or "L0"
        desc = RISK_LEVEL_DEFINITIONS.get(lvl, "")
        print(f" - {lvl} [{r['cnt']:>3} 件] : {desc}")
    print("=" * 78 + "\n")

    conn.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="模块 3：大模型搞钱赛道提炼与四级合规风险标注流水线")
    parser.add_argument("--force", action="store_true", help="强制对全库重新标注覆盖")
    parser.add_argument("--rule-engine", action="store_true", help="直接启用本地高保真确定性规则引擎，免API调用")
    args = parser.parse_args()

    if args.rule_engine:
        # 清空 channels 强制走 rule engine
        MultiChannelLLMClient.__init__ = lambda self: setattr(self, "channels", [])
        logger.info("已指定 --rule-engine 参数：全流程强制走本地确定性规则引擎")

    run_pipeline(force_reprocess=args.force)
