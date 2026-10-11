
import csv
import random

positive = []
negative = []

with open("steam_reviews_cleaned.csv", "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    columns = reader.fieldnames

    for row in reader:
        if row["label"] == "1":
            positive.append(row)
        elif row["label"] == "0":
            negative.append(row)

random.seed(42)

random.shuffle(positive)
random.shuffle(negative)

positive_split = int(len(positive) * 0.8)
negative_split = int(len(negative) * 0.8)

training = positive[:positive_split] + negative[:negative_split]
validation = positive[positive_split:] + negative[negative_split:]

random.shuffle(training)
random.shuffle(validation)

with open("train.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(training)

with open("validation.csv", "w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()
    writer.writerows(validation)

print("Training reviews:", len(training))
print("Validation reviews:", len(validation))
print("Total reviews:", len(training) + len(validation))
