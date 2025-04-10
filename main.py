# main.py

# --- Fix import paths so 'agents' works from any location ---
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

# --- Import AI agents ---
from summary_action_agent import get_summary_and_actions
from routing_agent import route_ticket
from recommendation_agent import build_ticket_index, recommend_solution
from time_estimation_agent import train_time_estimator, estimate_resolution_time

# Load your data file
CSV_PATH = "/Users/jeffrinjijo/Desktop/ai_customer_support /Dataset/[Usecase 7] AI-Driven Customer Support Enhancing Efficiency Through Multiagents​/Historical_ticket_data.csv"

# --- Simulated user ticket input ---
user_input = """
Hi, my app keeps crashing every time I try to pair it with an older smart bulb. 
It says "device not supported" even though it worked fine on version 4.8. 
Please fix this ASAP — I need it for my automated routines!
"""

# --- Agent 1: Summarize and extract actions ---
summary_result = get_summary_and_actions(user_input)
print("📄 SUMMARY + ACTIONS:")
print(summary_result)
print("--------------------------------------------------")

# --- Agent 2: Route to team ---
routing_result = route_ticket(priority="HIGH", sentiment="FRUSTRATED", category="DEVICE COMPATIBILITY ERROR")
print("🚚 ROUTING DECISION:")
print(f"➡️ Assign to: {routing_result}")
print("--------------------------------------------------")

# --- Agent 3: Recommend resolution ---
index, ticket_texts, df = build_ticket_index(CSV_PATH)
matched, solution = recommend_solution("Device compatibility error with smart bulb", index, ticket_texts, df)
print("💡 RESOLUTION RECOMMENDATION:")
print(f"🧾 Most similar ticket: {matched}")
print(f"✅ Suggested fix: {solution}")
print("--------------------------------------------------")

# --- Agent 4: Estimate resolution time ---
train_time_estimator(CSV_PATH)
estimated_time = estimate_resolution_time(
    priority="HIGH",
    sentiment="FRUSTRATED",
    category="DEVICE COMPATIBILITY ERROR"
)
print("⏱️ RESOLUTION TIME ESTIMATION:")
print(f"🕒 Expected to resolve in: {estimated_time} hours")
print("--------------------------------------------------")
