import os
from dotenv import load_dotenv
from google import genai
from PIL import Image

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

client = genai.Client(api_key=API_KEY)

SAVE_FOLDER = "generated_food_images"
os.makedirs(SAVE_FOLDER, exist_ok=True)


item_names = [
"Aloo Tikki","Bombil Fry","Brownie","Bruschetta","Butter Chicken",
"Butter Roti","Buttermilk","Caesar Salad","Cheesecake","Chicken Fried Rice",
"Chicken Nihari","Chicken Wrap","Chilli Chicken","Chole Bhature","Club Sandwich",
"Coca Cola","Coconut Chutney","Cold Coffee","Coleslaw","Crab Masala",
"Crispy Corn","Dal Makhani","Date Pancake","Dim Sum","Double Ka Meetha",
"Egg Biryani","Filter Coffee","Fish Curry","Fish Thali","Fish and Chips",
"French Fries","Fried Papad","Garlic Bread","Garlic Naan","Green Chutney",
"Grilled Chicken Burger","Gulab Jamun","Honey Noodles","Hyderabadi Chicken Biryani",
"Hyderabadi Mutton Biryani","Iced Tea","Idli Sambar","Jalebi","Jaljeera",
"Kadai Paneer","Kathi Roll","Kheer","Kokum Sharbat","Lassi",
"Lemon Iced Tea","Lemonade","Lucknowi Biryani","Margherita Pizza","Masala Chai",
"Masala Dosa","Masala Soda","Medu Vada","Milkshake","Mirchi Ka Salan",
"Modak","Mughlai Paratha","Mutton Korma","Mutton Rogan Josh","Mysore Pak",
"Naan","Nimbu Pani","Onion Raita","Onion Rings","Onion Salad",
"Paneer Tikka Masala","Pani Puri","Panna Cotta","Papad","Pasta Alfredo",
"Pav Bhaji","Payasam","Penne Arrabiata","Pepperoni Pizza","Phirni",
"Podi","Pongal","Potato Wedges","Prawn Masala","Puran Poli",
"Qubani Ka Meetha","Rabri","Raita","Rajma Chawal","Rasam",
"Rasmalai","Rava Dosa","Rooh Afza","Rooh Afza Sharbat","Roomali Roti",
"Salan","Sambar","Samosa","Schezwan Noodles","Sev",
"Shahi Paneer","Shahi Tukda","Sheermal","Sol Kadhi","Spring Roll",
"Steamed Rice","Sugarcane Juice","Tamarind Chutney","Tiramisu","Tomato Chutney",
"Uttapam","Vada Pav","Veg Biryani","Veg Burger","Veg Hakka Noodles",
"Veg Manchurian","Virgin Mojito"
]


def generate_food_image(item):

    prompt = f"""
High-end food delivery app product photo of {item}.

Professional food photography, centered dish, isolated on clean neutral background,
soft studio lighting, ultra realistic textures, vibrant colors,
restaurant quality plating, slight top-down angle.

No text, no logos, no packaging, no watermark.

Square composition, 4k food photography.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt
        )

        MIME_TO_EXT = {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/jpg": ".jpg",
            "image/webp": ".webp",
            "image/gif": ".gif",
        }

        for part in response.candidates[0].content.parts:

            if part.inline_data:

                image_bytes = part.inline_data.data
                mime = getattr(part.inline_data, "mime_type", None) or "image/png"
                ext = MIME_TO_EXT.get(mime, ".png")

                file_path = os.path.join(
                    SAVE_FOLDER,
                    item.replace(" ", "_") + ext
                )

                with open(file_path, "wb") as f:
                    f.write(image_bytes)

                print("Saved:", file_path)

    except Exception as e:
        print("Failed:", item, e)


for item in item_names:

    print("Generating image for:", item)
    generate_food_image(item)

print("\nAll images generated.")