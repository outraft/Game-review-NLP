import csv


def check_labels(input_file):
    labels = {}
    conflicts = []

    with open(input_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            comment = row["comment"].strip().lower()
            label = row["label"].strip()

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

    corrections = {
        "loved this game": "1",
        "best game ever! 🤩": "1",
        "i hate this.": "0"
    }

    empty_count = 0
    duplicate_count = 0
    meaningless_count = 0
    conflict_count = 0
    corrected_count = 0

    with open(input_file, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = reader.fieldnames

        for row in reader:
            comment = row["comment"].strip()
            lower_comment = comment.lower()

            if comment == "":
                empty_count += 1
                continue

            if lower_comment in conflicts:
                conflict_count += 1
                continue

            if lower_comment in comments:
                duplicate_count += 1
                continue

            if comment.isdigit():
                meaningless_count += 1
                continue

            if lower_comment in meaningless:
                meaningless_count += 1
                continue

            if comment in punctuation:
                meaningless_count += 1
                continue

            if all(not character.isalnum() for character in comment):
                meaningless_count += 1
                continue

            row["comment"] = comment

            if lower_comment in corrections:
                new_label = corrections[lower_comment]

                if row["label"] != new_label:
                    corrected_count += 1

                row["label"] = new_label

            if row["label"] == "1":
                row["positive or negative"] = "positive"

            elif row["label"] == "0":
                row["positive or negative"] = "negative"

            else:
                continue

            reviews.append(row)
            comments.append(lower_comment)

    with open(output_file, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(reviews)

    print("Cleaning completed.")
    print("Final number of reviews:", len(reviews))
    print("Empty reviews removed:", empty_count)
    print("Duplicate reviews removed:", duplicate_count)
    print("Meaningless reviews removed:", meaningless_count)
    print("Conflicting review rows removed:", conflict_count)
    print("Incorrect labels corrected:", corrected_count)


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
    print("Total reviews:", positive + negative)


conflicts = check_labels("steam_reviews_raw.csv")

clean_reviews(
    "steam_reviews_raw.csv",
    "steam_reviews_cleaned.csv",
    conflicts
)

count_labels("steam_reviews_cleaned.csv")