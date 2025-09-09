"""======STUDENT GRADER======"""
sub = int(input("Enter number of subjects: "))
marks = {}

for i in range(sub):
    sub_name = input(f"Enter subject name: ")
    mark = float(input(f"Enter marks for {sub_name}: "))
    marks[sub_name] = mark

def cal_grade(marks):
    total = sum(marks.values())
    avge = total / len(marks)

    if avge >= 90:
        grade = "A+"
    elif avge >= 80:
        grade = "A"
    elif avge >= 70:
        grade = "B"
    elif avge >= 60:
        grade = "C"
    elif avge >= 50:
        grade = "D"
    else:
        grade = "F"

    return total, avge, grade

total, avge, grade = cal_grade(marks)

print("---- Student Report ----")
print(f"Total Marks: {total}")
print(f"Average Marks: {avge:.2f}")
print(f"Grade: {grade}")
