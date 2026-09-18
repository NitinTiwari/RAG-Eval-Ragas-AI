"""
eval_dataset.py
Hand-written evaluation set: questions paired with ground-truth
answers, used as the "ground_truth" column for RAGAS metrics
like context_recall and answer_correctness.
"""

EVAL_QUESTIONS = [
    {
        "question": "How many vacation days do employees get in their first two years?",
        "ground_truth": "Employees accrue 18 days of paid vacation per year during their first two years of employment.",
    },
    {
        "question": "Which days are mandatory in-office days under the hybrid work policy?",
        "ground_truth": "Tuesday and Thursday are the mandatory in-office days under Acme Corp's hybrid work model.",
    },
    {
        "question": "What percentage of the health insurance premium does Acme Corp cover for dependents?",
        "ground_truth": "Acme Corp covers 60% of the health insurance premium for dependents.",
    },
    {
        "question": "What is the price of the Growth plan and how many shipments does it include?",
        "ground_truth": "The Growth plan costs $499 per month and includes up to 5,000 shipments.",
    },
    {
        "question": "What security certifications does the Acme Logistics Platform hold?",
        "ground_truth": "Acme is SOC 2 Type II certified and ISO 27001 compliant.",
    },
    {
        "question": "How much can employees be reimbursed for home office equipment each year?",
        "ground_truth": "Employees can be reimbursed up to $500 per year for home office equipment.",
    },
]
