import io
import sys
from PIL import Image
from argparse import ArgumentParser
from contact_recognizer import recognize_contact
from card_recognizer import detect_contours
from text_recognizer import get_text

def main():
    parser = ArgumentParser()
    parser.add_argument("--path", "-p", help="Full path to the business card image file. Run 'app.py -p './test_data/card.jpg' for test.'")
    args = parser.parse_args()
    
    if(len(sys.argv)<2):
        raise ValueError("Program doesn't run with zero arguments. Please see help 'app.py --help' to see possible arguments.")

    file_path = args.path
    with open(file_path, "rb") as image:
        image_data = image.read()
        pil_img = Image.open(io.BytesIO(image_data)).convert('RGB') 
        
    print('Image processing can take sometime. Please wait...')
    business_card =  detect_contours(pil_img)
    text = get_text(business_card)
    contact = recognize_contact(text)
    contact.print_contact()

if __name__ == "__main__":
    main()

