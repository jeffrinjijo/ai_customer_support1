# agents/routing_agent.py

def route_ticket(priority, sentiment, category):
    """
    Determines which team to route the ticket to based on priority, sentiment, and category.
    """

    priority = priority.strip().upper()
    sentiment = sentiment.strip().upper()
    category = category.strip().upper()

    # Routing logic (customize this as needed)
    if priority == "CRITICAL" or sentiment in ["URGENT", "FRUSTRATED"]:
        return "TECHNICAL_SUPPORT_TEAM"
    elif "PAYMENT" in category:
        return "BILLING_TEAM"
    elif "ACCOUNT" in category:
        return "ACCOUNT_MANAGEMENT_TEAM"
    elif sentiment in ["CONFUSED", "ANXIOUS"]:
        return "CUSTOMER_CARE_TEAM"
    else:
        return "GENERAL_SUPPORT_TEAM"
