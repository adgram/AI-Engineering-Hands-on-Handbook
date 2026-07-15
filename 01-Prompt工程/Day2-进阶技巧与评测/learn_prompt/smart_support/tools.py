import sys, json
from pathlib import Path
common_path = Path(__file__).parent.parent.parent.parent.parent / "common"
sys.path.insert(0, str(common_path))
from llm_client import LLMClient

client = LLMClient()


def search_orders(customer_id: str = "", status: str = "", date_range: str = "") -> str:
    orders = [
        {"id": "ORD-2024-001", "status": "已发货", "product": "无线耳机", "amount": 299, "date": "2024-12-01"},
        {"id": "ORD-2024-002", "status": "处理中", "product": "蓝牙音箱", "amount": 599, "date": "2024-12-05"},
        {"id": "ORD-2024-003", "status": "已取消", "product": "充电宝", "amount": 129, "date": "2024-11-28"},
        {"id": "ORD-2024-004", "status": "已发货", "product": "手机壳", "amount": 39, "date": "2024-12-10"},
    ]
    if status:
        orders = [o for o in orders if o["status"] == status]
    return json.dumps(orders, ensure_ascii=False)


def track_delivery(order_id: str = "") -> str:
    tracking = {
        "ORD-2024-001": {"status": "运输中", "location": "上海分拣中心", "estimated": "2024-12-03"},
        "ORD-2024-004": {"status": "已签收", "location": "北京朝阳区", "signed": "2024-12-12"},
    }
    result = tracking.get(order_id, {"error": f"订单 {order_id} 无物流信息"})
    return json.dumps(result, ensure_ascii=False)


def cancel_order(order_id: str = "", reason: str = "") -> str:
    valid_orders = ["ORD-2024-002", "ORD-2024-003"]
    if order_id in valid_orders:
        return json.dumps(
            {"success": True, "message": f"订单 {order_id} 已取消", "refund": "原路退回"},
            ensure_ascii=False,
        )
    return json.dumps(
        {"success": False, "message": f"订单 {order_id} 无法取消（可能已发货）"},
        ensure_ascii=False,
    )


def check_return_policy(product_category: str = "") -> str:
    policies = {
        "电子产品": "7天无理由退货，需保持包装完好",
        "日用品": "15天无理由退货",
        "食品": "不支持退货（质量问题除外）",
    }
    result = policies.get(product_category, "请提供商品类别")
    return json.dumps({"category": product_category, "policy": result}, ensure_ascii=False)


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_orders",
            "description": "查询订单信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "客户ID"},
                    "status": {
                        "type": "string",
                        "enum": ["已发货", "处理中", "已取消", ""],
                        "description": "订单状态",
                    },
                    "date_range": {"type": "string", "description": "日期范围，如 2024-12-01~2024-12-31"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_delivery",
            "description": "查询物流信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单ID"}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "取消订单",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "订单ID"},
                    "reason": {"type": "string", "description": "取消原因"},
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_policy",
            "description": "查询退货政策",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_category": {"type": "string", "description": "商品类别"}
                },
                "required": ["product_category"],
            },
        },
    },
]

TOOL_HANDLERS = {
    "search_orders": search_orders,
    "track_delivery": track_delivery,
    "cancel_order": cancel_order,
    "check_return_policy": check_return_policy,
}
