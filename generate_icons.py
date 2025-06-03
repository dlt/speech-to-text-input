#!/usr/bin/env python3
"""
Generate icon images for the system tray application
"""

from PIL import Image, ImageDraw
import os


def create_microphone_icon(size=64, color='black', background=None):
    """Create a microphone icon"""
    # Create an image
    image = Image.new('RGBA', (size, size), background or (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    
    # Scale factors
    scale = size / 64
    
    # Color definitions
    colors = {
        'black': (0, 0, 0, 255),
        'gray': (128, 128, 128, 255),
        'red': (255, 0, 0, 255),
        'green': (0, 200, 0, 255),
        'orange': (255, 165, 0, 255),
        'white': (255, 255, 255, 255)
    }
    
    mic_color = colors.get(color, colors['black'])
    
    # Draw microphone body (scaled)
    draw.ellipse([
        int(20 * scale), int(10 * scale),
        int(44 * scale), int(40 * scale)
    ], fill=mic_color, outline=mic_color)
    
    draw.rectangle([
        int(20 * scale), int(25 * scale),
        int(44 * scale), int(35 * scale)
    ], fill=mic_color, outline=mic_color)
    
    # Draw microphone stand
    draw.rectangle([
        int(30 * scale), int(35 * scale),
        int(34 * scale), int(50 * scale)
    ], fill=mic_color, outline=mic_color)
    
    draw.rectangle([
        int(24 * scale), int(50 * scale),
        int(40 * scale), int(54 * scale)
    ], fill=mic_color, outline=mic_color)
    
    # Draw arc for microphone holder
    draw.arc([
        int(15 * scale), int(30 * scale),
        int(49 * scale), int(50 * scale)
    ], start=30, end=150, fill=mic_color, width=int(3 * scale))
    
    return image


def main():
    """Generate icon files"""
    # Create icons directory
    os.makedirs('icons', exist_ok=True)
    
    # Generate icons in different states and sizes
    states = {
        'ready': 'green',
        'paused': 'gray',
        'recording': 'red',
        'processing': 'orange',
        'default': 'black'
    }
    
    sizes = [16, 32, 64, 128, 256]
    
    for state, color in states.items():
        for size in sizes:
            # Regular icon
            icon = create_microphone_icon(size, color)
            icon.save(f'icons/mic_{state}_{size}.png')
            
            # Also create @2x versions for macOS
            if size <= 64:
                icon_2x = create_microphone_icon(size * 2, color)
                icon_2x.save(f'icons/mic_{state}_{size}@2x.png')
    
    # Create a template icon for macOS (black on transparent)
    template = create_microphone_icon(64, 'black')
    template.save('icons/mic_template.png')
    
    # Create white version for dark mode
    white_icon = create_microphone_icon(64, 'white')
    white_icon.save('icons/mic_white.png')
    
    print("✅ Icons generated in 'icons' directory")


if __name__ == "__main__":
    main()