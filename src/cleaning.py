
import csv


def check_labels(input_file):
    labels = {}
    conflicts = []

    with open(input_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            comment = row["comment"].strip().lower()
            label = row["label"]

            if comment in labels:
                if labels[comment] != label:
                    if comment not in conflicts:
                        conflicts.append(comment)
            else:
                labels[comment] = label

    with open("conflicting_reviews.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["comment"])
        
        for comment in conflicts:
            writer.writerow([comment])

    print("Conflicting reviews:", len(conflicts))

    return conflicts


def clean_reviews(input_file, output_file, conflicts):
    reviews = []
    comments = []

    meaningless = [
        "jh", "h", "o", "n", "x", "l", "m",
        "d", "s", "z", "q", "g", "y", "f",
        "j", "n/a", "+++", ".q", ".!.", "ㅤ",
        "fht6u", "ed3wq", "wdawd", "dsfds",
        "kljh", "drff", "asds", "fddfff",
        "jbyfkb", "dddd", "eere", "gfdgfd",
        "yjtyh", "sddsad", "gfgf",
        "....", "******", "‎"
    ]

    punctuation = [".", "..", ",", "!", "?", ";", "،"]

    with open(input_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        for row in reader:
            comment = row["comment"].strip()
            lower_comment = comment.lower()

            if comment == "":
                continue

            if lower_comment in comments:
                continue

            if comment.isdigit():
                continue

            if lower_comment in meaningless:
                continue

            if comment in punctuation:
                continue

            if lower_comment in conflicts:
                continue

            row["comment"] = comment
            reviews.append(row)
            comments.append(lower_comment)

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(reviews)

    print("Cleaning completed.")
    print("Final number of reviews:", len(reviews))


def count_labels(input_file):
    positive = 0
    negative = 0

    with open(input_file, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            if row["label"] == "1":
                positive += 1
            elif row["label"] == "0":
                negative += 1

    print("Positive reviews:", positive)
    print("Negative reviews:", negative)


conflicts = check_labels("steam_reviews_raw.csv")

clean_reviews(
    "steam_reviews_raw.csv",
    "steam_reviews_cleaned.csv",
    conflicts
)

count_labels("steam_reviews_cleaned.csv")
