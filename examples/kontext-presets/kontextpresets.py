import json


class LoadKontextPresets:
    data = {
        "prefix": "You are a creative prompt engineer. Your mission is to analyze the provided image and generate exactly 1 distinct image transformation *instructions*.",
        "presets": [
            {
                "name": "情境深度融合",
                "brief": "The provided image is a composite with a head and body from drastically different contexts (lighting, style, condition). Your mission is to generate instructions for a complete narrative and physical transformation of the head to flawlessly match the body and scene. The instructions must guide the AI to: 1. **Cinematic Re-Lighting**: Describe in vivid detail the scene's light sources (color, direction, hardness) and how this light should sculpt the head's features with new, appropriate shadows and highlights. 2. **Contextual Storytelling**: Instruct to add physical evidence of the scene's story onto the head, such as grime from a battle, sweat from exertion, or rain droplets from a storm. 3. **Color Grading Unification**: Detail how to apply the scene's specific color grade (e.g., cool desaturated tones, warm golden hour hues) to the head. 4. **Asset & Hair Adaptation**: Command the modification or removal of out-of-place elements (like clean jewelry in a gritty scene) and the restyling of the hair to fit the environment (e.g., messy, windblown, wet). 5. **Flawless Final Integration**: As the final step, describe the process of blending the neckline to be completely invisible, ensuring a uniform film grain and texture across the entire person.",
            },
            {
                "name": "无痕融合",
                "brief": "This image is a composite with minor inconsistencies between the head and body. Your task is to generate instructions for a subtle but master-level integration. Focus on creating a photorealistic and utterly convincing final image. The instructions should detail: 1. **Micro-Lighting Adjustment**: Fine-tune the lighting and shadows around the neck and jawline to create a perfect match. 2. **Skin Tone & Texture Unification**: Describe the process of unifying the skin tones for a seamless look, and more importantly, harmonizing the micro-textures like pores, fine hairs, and film grain across the blended area. 3. **Edge Blending Perfection**: Detail how to create an invisible transition at the neckline, making it appear as if it was never separate.",
            },
            {
                "name": "场景传送",
                "brief": "Imagine the main subject of the image is suddenly teleported to a completely different and unexpected environment, while maintaining their exact pose. Your instruction should describe this new, richly detailed scene. For example, a person in a business suit is now standing in the middle of an enchanted, glowing forest, or a beachgoer is now on a futuristic spaceship bridge. The instruction must detail how the new environment's lighting and atmosphere should realistically affect the subject.",
            },
            {
                "name": "移动镜头",
                "brief": "Propose a dramatic and purposeful camera movement that reveals a new perspective or emotion in the scene. Instead of a simple change, describe the *type* of shot and its *narrative purpose*. For example, 'Change to a dramatic low-angle shot from the ground, making the subject appear heroic and monumental against the sky,' or 'Switch to a dizzying Dutch angle shot, tilting the horizon to create a sense of unease and disorientation'.",
            },
            {
                "name": "重新布光",
                "brief": "Completely transform the mood and story of the image by proposing a new, cinematic lighting scheme. Describe a specific lighting style linked to a genre or mood. For example: 'Relight the scene with Film Noir aesthetics: hard, single-source key light creating deep shadows and high contrast,' or 'Relight with the warmth of a magical Golden Hour: soft, golden light from a low angle, wrapping around the subject and creating long, gentle shadows.'",
            },
            {
                "name": "专业产品图",
                "brief": "Re-imagine this image as a high-end commercial product photograph for a luxury catalog. Describe a clean, professional setting. The instruction should specify: 1. **Studio Lighting**: Detail a sophisticated lighting setup (e.g., three-point lighting with a softbox key light, a rim light for separation, and a fill light to soften shadows). 2. **Composition**: Describe a clean, minimalist composition on a seamless background or in a luxury lifestyle setting. 3. **Product Focus**: Emphasize capturing crisp details, perfect textures, and an aspirational mood.",
            },
            {
                "name": "画面缩放",
                "brief": "Describe a specific zoom action that serves a narrative purpose. Propose either: 1. **A dramatic 'push-in' (zoom in)**: 'Slowly push in on the subject's eyes to reveal a subtle, hidden emotion.' or 2. **A revealing 'pull-out' (zoom out)**: 'Pull the camera back to reveal a surprising or vast new element in the environment that re-contextualizes the subject's situation.'",
            },
            {
                "name": "图像上色",
                "brief": "Describe a specific artistic style for colorizing a black and white image. Go beyond simple colorization. For example: 'Colorize this image with the vibrant, high-contrast, and slightly surreal palette of the Technicolor films from the 1950s,' or 'Apply a muted, melancholic color palette with desaturated blues and earthy tones, reminiscent of a modern independent film.'",
            },
            {
                "name": "电影海报",
                "brief": "Transform the image into a compelling movie poster for a specific, imagined film genre. Describe the full poster concept: 1. **Genre & Title**: Invent a movie title and genre (e.g., Sci-Fi Thriller: 'ECHOES OF TOMORROW'). 2. **Visual Style**: Describe how to treat the image (e.g., 'apply a gritty, high-contrast filter'). 3. **Typography & Text**: Instruct to add a stylized title, a dramatic tagline (e.g., 'The future is listening.'), and other elements like actor names and release date.",
            },
            {
                "name": "卡通漫画化",
                "brief": "Redraw the entire image in a specific, iconic animated or illustrated style. Be descriptive about the chosen style. For example: 'Convert the image into the style of a 1990s Japanese anime cel, characterized by sharp character outlines, expressive eyes, and hand-painted backgrounds,' or 'Re-imagine the scene in the visual language of a modern Pixar film, with soft, appealing shapes, detailed textures, and warm, bounce lighting.'",
            },
            {
                "name": "移除文字",
                "brief": "Describe the task of removing all text from the image as a meticulous restoration project. 'Carefully and seamlessly remove all text, logos, and lettering from the image. The goal is to reconstruct the underlying surfaces and textures so perfectly that there is no hint that text ever existed there.'",
            },
            {
                "name": "更换发型",
                "brief": "Describe a complete hair transformation for the subject that tells a story or embodies a new persona. Be specific about the style, color, and texture. For example: 'Give the subject a bold, punk-inspired pixie cut with a vibrant magenta color, featuring a choppy, textured finish,' or 'Transform her hair into long, elegant, Pre-Raphaelite waves with a deep auburn hue, appearing soft and voluminous.'",
            },
            {
                "name": "肌肉猛男化",
                "brief": "Dramatically transform the subject into a hyper-realistic, massively muscled bodybuilder, as if they are a character from a fantasy action movie. Describe the transformation in detail: 'Exaggerate and define every muscle group—biceps, triceps, pectorals, and abdominals—to heroic proportions. The skin should be taut over the muscles, with visible veins. Modify the clothing, perhaps tearing it, to accommodate and reveal the new, powerful physique.'",
            },
            {
                "name": "清空家具",
                "brief": "Imagine the room in the image has been completely emptied for a renovation. Instruct the AI to meticulously remove all furniture, appliances, decorations, carpets, and even light fixtures from the ceiling and walls. The instruction should state that the goal is to reveal the room's 'bare bones'—the empty floor, walls, and ceiling, paying close attention to realistically recreating the surfaces that were previously hidden.",
            },
            {
                "name": "室内设计",
                "brief": "You are a world-class interior designer tasked with redesigning this space in a specific, evocative style. While keeping the room's core structure (walls, windows, doors) intact, describe a complete redesign concept. For example: 'Redesign this room with a 'Japandi' (Japanese-Scandinavian) aesthetic: light wood furniture with clean lines, a neutral color palette of beige and gray, minimalist decor, and soft, diffused natural lighting.'",
            },
            {
                "name": "季节变换",
                "brief": "Transform the entire scene to be convincingly set in a different season, focusing on atmospheric and sensory details. Propose a season and describe its effects. For example: 'Plunge the scene into a deep, quiet Autumn. Change all foliage to rich tones of crimson and gold. Litter the ground with fallen leaves. The air should feel crisp, and the light should have a low, golden quality. Adjust the subject's clothing to include a cozy sweater or light jacket.'",
            },
            {
                "name": "时光旅人",
                "brief": "Visibly and realistically age or de-age the main subject, as if we are seeing them at a different stage of their life. For an older version, describe 'adding fine lines around the eyes and mouth, silver strands woven into the hair, and the subtle wisdom in their expression.' For a younger version, describe 'smoothing the skin to a youthful glow, restoring the hair's original vibrant color, and capturing a sense of bright-eyed optimism'.",
            },
            {
                "name": "材质置换",
                "brief": "Re-imagine the main subject as a masterfully crafted sculpture made from an unexpected material, keeping their pose intact. Describe the new material's properties in detail. For example: 'Transform the subject into a statue carved from a single piece of dark, polished obsidian. Describe its glossy, reflective surface, how it catches the light, and the subtle, natural imperfections within the stone.'",
            },
            {
                "name": "微缩世界",
                "brief": "Convert the entire scene into a charming and highly detailed miniature model world, as if viewed through a tilt-shift lens. Describe the visual effects needed: 'Apply a very shallow depth of field to blur the top and bottom of the image, making the scene look tiny. Sharpen the details and boost the color saturation to enhance the artificial, toy-like appearance of all subjects and objects.'",
            },
            {
                "name": "幻想领域",
                "brief": "Transport the entire scene and its subject into a specific, richly detailed fantasy or sci-fi universe. Describe the complete aesthetic overhaul. For example: 'Re-imagine the scene in a Steampunk universe. All modern technology is replaced with intricate brass and copper machinery, full of gears and steam pipes. The subject's clothing is transformed into Victorian-era attire with leather and brass accessories.'",
            },
            {
                "name": "衣橱改造",
                "brief": "Give the subject a complete fashion makeover into a specific, well-defined style, keeping the background and pose the same. Describe the entire outfit in detail. For example: 'Dress the subject in a 'Dark Academia' aesthetic: a tweed blazer, a dark turtleneck sweater, tailored trousers, and leather brogues. Add an accessory like a vintage leather satchel or a pair of classic spectacles.'",
            },
            {
                "name": "艺术风格模仿",
                "brief": "Repaint the entire image in the unmistakable style of a famous art movement. Be descriptive and evocative. For example: 'Transform the image into a Post-Impressionist painting in the style of Van Gogh, using thick, swirling brushstrokes (impasto), vibrant, emotional colors, and a dynamic sense of energy and movement in every element.'",
            },
            {
                "name": "蓝图视角",
                "brief": "Convert the image into a detailed and aesthetically pleasing technical blueprint. Describe the visual transformation: 'Change the background to a classic cyanotype-blue. Redraw the main subject and key objects using clean, white, schematic outlines. Overlay the image with fictional measurement lines, annotations, and technical callouts to complete the architectural drawing effect.'",
            },
            {
                "name": "添加倒影",
                "brief": "Introduce a new, reflective surface into the scene to create a more dynamic and compelling composition. Describe the placement and quality of the reflection. For example: 'After a rain shower, the street is now covered in a thin layer of water, creating a stunning, mirror-like reflection of the subject and the moody, illuminated sky above. The reflection should be slightly distorted by ripples in the water.'",
            },
            {
                "name": "像素艺术",
                "brief": "Deconstruct the image into the charming, retro aesthetic of 16-bit era pixel art. Describe the technical and artistic constraints: 'Reduce the entire image to a limited color palette of 64 colors. Redraw all shapes and subjects with sharp, aliased pixel edges. Use dithering patterns to create gradients and shading, capturing the authentic look and feel of a classic Super Nintendo game.'",
            },
            {
                "name": "铅笔手绘",
                "brief": "You are a master of pencil sketching. Write prompt words to transform the picture into a pencil sketch style based on the image content. The description needs to be precise and detailed, and avoid describing content unrelated to the task.",
            },
            {
                "name": "油画风格",
                "brief": "Transform the provided image into an oil painting style. Describe the brushwork, color palette, and overall mood to achieve an authentic oil - painting aesthetic. For example, use thick, impasto brushstrokes to add texture, choose a rich and vibrant color palette typical of oil paintings, and create a moody or vivid atmosphere according to the scene.",
            },
        ],
        "suffix": "Your response must consist of concise instruction ready for the image editing AI. Do not add any conversational text, explanations, or deviations; only the instructions.",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "preset": ([preset["name"] for preset in cls.data.get("presets", [])],),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("Prompt",)
    FUNCTION = "get_preset"
    CATEGORY = "utils"

    @classmethod
    def get_brief_by_name(cls, name):
        for preset in cls.data.get("presets", []):
            if preset["name"] == name:
                return preset["brief"]
        return None

    def get_preset(cls, preset):
        # We need to find the English name that corresponds to the selected Chinese name
        english_name = ""
        for p in cls.data.get("presets", []):
            if p["name"] == preset:
                # This seems redundant, but it's a way to ensure we get the brief for the selected item
                # The logic in the original file would work fine, but this makes it explicit.
                brief_text = p["brief"]
                break

        brief = "The Brief:" + brief_text
        fullString = (
            cls.data.get("prefix") + "\n" + brief + "\n" + cls.data.get("suffix")
        )
        return (fullString,)


NODE_CLASS_MAPPINGS = {
    "LoadKontextPresets": LoadKontextPresets,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadKontextPresets": "Kontext Presets (中文版)",
}
