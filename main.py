from src import SNE1A, SNEII

def main():
    try:
        # Ask the user which supernova they want to analyze
        while(True):
            choice = input("Which supernova would you like to analyze? (Type 1 for SNE1A, Type 2 for SNEII): ")
            
            # 1 = SNE1A, 2 = SNEII
            if choice == '1':
                SNE1A.main()
                break
            elif choice == '2':
                SNEII.main()
                break
            else:
                print("Invalid choice. Please type 1 for SNE1A or 2 for SNEII.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()