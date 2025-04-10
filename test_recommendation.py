# test_recommendation.py

from agents.recommendation_agent import build_ticket_index, recommend_solution

index, ticket_texts, df = build_ticket_index("/Users/jeffrinjijo/Desktop/ai_customer_support /Dataset/[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents​/Historical_ticket_data.csv")

test_summary = "Device compatibility error with older smart thermostat"

matched_summary, solution = recommend_solution(test_summary, index, ticket_texts, df)

print("🔍 Most Similar Ticket Category:", matched_summary)
print("✅ Recommended Resolution:", solution)
