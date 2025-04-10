from agents.time_estimation_agent import train_time_estimator, estimate_resolution_time

model = train_time_estimator("/Users/jeffrinjijo/Desktop/ai_customer_support /Dataset/[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents​/Historical_ticket_data.csv")

predicted_time = estimate_resolution_time(
    priority="HIGH",
    sentiment="FRUSTRATED",
    category="DEVICE COMPATIBILITY ERROR"
)

print(f"⏱️ Estimated Resolution Time: {predicted_time} hours")
