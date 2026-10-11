
import csv


def clean_reviews(input_file, output_file):
    reviews = []
    comments = []

    meaningless = [
        "jh", "h", "o", "n", "x", "l", "m",
        "d", "s", "z", "q", "g", "y", "f",
        "j", "n/a", "+++", ".q", ".!.", "ㅤ"
    ]

    with open(input_file, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        for row in reader:
            comment = row["comment"].strip()
            lower_comment = comment.lower()

            if comment != "" and lower_comment not in comments:
                if not comment.isdigit():
                    if lower_comment not in meaningless:
                        if comment not in [".", "..", ",", "!", "?", ";", "،"]:
                            row["comment"] = comment
                            reviews.append(row)
                            comments.append(lower_comment)

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(reviews)

    print("Cleaning completed.")
    print("Reviews after cleaning:", len(reviews))


def check_labels(input_file):
    labels = {}
    conflicts = []

    with open(input_file, "r", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            comment = row["comment"].strip().lower()
            label = row["label"]

            if comment in labels:
                if labels[comment] != label:
                    conflicts.append({
                        "comment": comment,
                        "first_label": labels[comment],
                        "different_label": label
                    })
            else:
                labels[comment] = label

    with open("conflicting_reviews.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["comment", "first_label", "different_label"]
        )
        writer.writeheader()
        writer.writerows(conflicts)

    print("Conflicting reviews:", len(conflicts))

    return conflicts


def remove_conflicts(input_file, conflicts):
    conflicting_comments = []

    for row in conflicts:
        if row["comment"] not in conflicting_comments:
            conflicting_comments.append(row["comment"])

    final_reviews = []
    positive = 0
    negative = 0

    with open(input_file, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        for row in reader:
            comment = row["comment"].strip().lower()

            if comment not in conflicting_comments:
                final_reviews.append(row)

                if row["label"] == "1":
                    positive += 1
                elif row["label"] == "0":
                    negative += 1

    with open(input_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(final_reviews)

    print("Final number of reviews:", len(final_reviews))
    print("Positive reviews:", positive)
    print("Negative reviews:", negative)


clean_reviews("steam_reviews_raw.csv", "steam_reviews_cleaned.csv")

conflicts = check_labels("steam_reviews_raw.csv")

remove_conflicts("steam_reviews_cleaned.csv", conflicts)
