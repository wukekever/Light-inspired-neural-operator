import colorsys

from PIL import Image, ImageDraw, ImageFont

text = "Let There Be Light: Reflection, Refraction and Scattering for Neural Operators of Parametric PDEs"
font = ImageFont.truetype(
    "DejaVuSans.ttf", 18.5
)  # Use a larger font for better visibility

# Create canvas
img = Image.new("RGBA", (1000, 60), (255, 255, 255, 0))
draw = ImageDraw.Draw(img)

# Get text width for gradient calculation
bbox = draw.textbbox((0, 0), text, font=font)
text_width = bbox[2] - bbox[0]

# Color each character (rainbow)
x = 20
for i, char in enumerate(text):
    hue = i / len(text)  # 0~1 mapped to rainbow color
    r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
    color = (int(r * 255), int(g * 255), int(b * 255))
    draw.text((x, 20), char, font=font, fill=color)
    char_bbox = draw.textbbox((0, 0), char, font=font)
    x += char_bbox[2] - char_bbox[0]

img.save("./doc/rainbow_title.png")
