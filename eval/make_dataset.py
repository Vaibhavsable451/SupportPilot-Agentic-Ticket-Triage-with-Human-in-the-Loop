"""Generate a synthetic labelled ticket dataset (replace with real tickets for production)."""
import random
import pandas as pd

random.seed(7)
BODY = {
    "billing": ["I was charged twice for my {plan} subscription", "Please explain the charge of ${amt} on my invoice",
                "My invoice shows the wrong amount", "How do I update the credit card on my billing profile",
                "I was billed after I cancelled my plan", "Can I get a receipt for my last payment"],
    "technical": ["The app crashes every time I open the {feat} page", "I get error {code} when I try to sync my data",
                  "The dashboard is not loading for me", "Push notifications stopped working on my phone",
                  "Exporting to CSV fails with a timeout", "The mobile app freezes on the login screen"],
    "account": ["I forgot my password and the reset email never arrives", "How do I change the email address on my account",
                "How can I enable two factor authentication", "My account is locked after too many login attempts",
                "I cannot log in to my account", "How do I update my profile name"],
    "shipping": ["Where is my order {order}", "My package {order} has not arrived after two weeks",
                 "I received the wrong item in order {order}", "Can I change the delivery address for {order}",
                 "The tracking for {order} has not updated in days", "When will order {order} be delivered"],
}
PRE = {"high": ["URGENT:", "This is urgent,", "I need help immediately.", "Critical issue:"],
       "medium": ["Hello,", "Hi team,", "Hi,", ""],
       "low": ["Just curious,", "When you have time,", "Quick question:", "No rush,"]}
POST = {"high": ["This is unacceptable.", "I am very frustrated.", "I need this fixed today."],
        "medium": ["Thanks.", "Please advise.", ""],
        "low": ["Thanks in advance!", "No hurry at all.", "Appreciate it."]}
rows = []
for cat, bodies in BODY.items():
    for _ in range(60):
        pr = random.choices(["high", "medium", "low"], [0.3, 0.45, 0.25])[0]
        body = random.choice(bodies).format(plan=random.choice(["Pro", "Team", "Basic"]), amt=random.choice([19, 49, 99]),
                                            feat=random.choice(["billing", "reports", "settings"]),
                                            code=random.choice(["E401", "E500", "E1002"]),
                                            order=f"ORD-{random.randint(1000, 1004)}")
        text = " ".join(x for x in [random.choice(PRE[pr]), body + ".", random.choice(POST[pr])] if x)
        rows.append({"text": text, "category": cat, "priority": pr})
pd.DataFrame(rows).sample(frac=1, random_state=1).to_csv("data/tickets.csv", index=False)
print(f"wrote {len(rows)} rows")
