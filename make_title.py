import colorsys

from PIL import Image, ImageDraw, ImageFont

text = "Light-inspired Neural Operator"
font = ImageFont.truetype(
    "DejaVuSans.ttf", 48
)  # Use a larger font for better visibility

# Create canvas
img = Image.new("RGBA", (1000, 80), (255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# Get text width for gradient calculation
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]

# Calculate starting x position to center text
x = (1000 - text_width) // 2

# Color each character (soft rainbow with low contrast)
for i, char in enumerate(text):
    hue = i / len(text)  # 0~1 mapped to rainbow color
    r, g, b = colorsys.hsv_to_rgb(
        hue, 0.6, 0.8
    )  # Reduced saturation and value for softer colors
    color = (int(r * 255), int(g * 255), int(b * 255))
    draw.text((x, 20), char, font=font, fill=color)
    char_bbox = draw.textbbox((0, 0), char, font=font)
    x += char_bbox[2] - char_bbox[0]

img.save("./doc/rainbow_title.png")
