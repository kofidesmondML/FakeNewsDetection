import subprocess

def main():
    scripts = [
        "data_processing.py",
        "clean_news_content.py",
        "features.py",
        "classifier.py"
    ]

    for script in scripts:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} completed.\n")

    print("All scripts have been executed successfully.")

if __name__ == "__main__":
    main()