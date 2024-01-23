import os
import csv

def app_setup():
    cwd = os.getcwd()
    file_name = f"{cwd}/data/persons.csv"

    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            print(row)
    

def main():
    app_setup()

if __name__ == "__main__":
    main()