"""Agent tool: mock order-lookup API (swap for your real OMS)."""
from langchain_core.tools import tool

ORDERS = {
    "ORD-1001": "Order ORD-1001 status: shipped via UPS, tracking 1Z999AA10123456784, estimated delivery 2026-10-16.",
    "ORD-1002": "Order ORD-1002 status: delivered on 2026-10-09 and left at the front door.",
    "ORD-1003": "Order ORD-1003 status: processing, expected to ship on 2026-10-15.",
    "ORD-1004": "Order ORD-1004 status: delayed by the carrier, new estimated delivery 2026-10-20.",
}


@tool
def lookup_order(order_id: str) -> str:
    """Look up the status of an order by its ID (format ORD-1234)."""
    return ORDERS.get(order_id.upper(), f"No order found with ID {order_id}.")
