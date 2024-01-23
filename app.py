import os

def app_setup():
    cwd = os.getcwd()
    print(f"{cwd}/data/contacts.csv")
    # read csv data
    

def main():
    app_setup()

if __name__ == "__main__":
    main()