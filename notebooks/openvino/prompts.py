from dataclasses import dataclass

@dataclass
class ExpDesc:
    prompt: str
    seed: int = 1
    negative_prompt: str = ''
    num_inference_steps: int = 20

DESCS = [
    ExpDesc(
        prompt="a portrait of an old coal miner in 19th century, beautiful painting with highly detailed face by greg rutkowskiand magali villanueve",
        negative_prompt="deformed face, Ugly, bad quality, lowres, monochrome, bad anatomy",
        seed=1507302932
    ),
    ExpDesc(
        prompt = "Pikachu commitingtax fraud, paperwork, exhausted, cute, really cute, cozy, by stevehanks, by lisa yuskavage, by serov valentin, by tarkovsky, 8 k render, detailed, cute cartoon style",
        seed = 345,
        negative_prompt="",
    ),
    ExpDesc(
        prompt = "amazon rainforest with many trees photorealistic detailed leaves",
        negative_prompt = "blurry, poor quality, deformed, cartoonish, painting",
        seed = 1137900754
    ),
    ExpDesc(
        prompt="autumn in paris, ornate, beautiful, atmosphere, vibe, mist, smoke, fire, chimney, rain, wet, pristine, puddles, melting, dripping, snow, creek, lush, ice, bridge, forest, roses, flowers, by stanley artgerm lau, greg rutkowski, thomas kindkade, alphonse mucha, loish, norman rockwell",
        negative_prompt="",
        seed = 2132889432
    ),
    ExpDesc(
        prompt="portrait of renaud sechan, pen and ink, intricate line drawings, by craig mullins, ruanjia, kentaro miura, greg rutkowski, loundraw",
        negative_prompt="hyperrealism",
        seed = 206890696,
    ),
    ExpDesc(
        prompt="An astronaut laying down in a bed of millions of vibrant, colorful flowers and plants, photoshoot",
        negative_prompt="deformed face, Ugly, bad quality, lowres, monochrome, bad anatomy",
        seed = 3997429436,
    ),
    ExpDesc(
        prompt="long range view, Beautiful Japanese flower garden, elegant bridges, waterfalls, pink and white, by Akihito Yoshida, Ismail Inceoglu, Karol Bak, Airbrush, Dramatic, Panorama, Cool ColorPalette, Megapixel, Lumen Reflections, insanely detailed and intricate, hypermaximalist, elegant, ornate, hyper realistic, super detailed, unreal engine",
        negative_prompt="lowres, bad, deformed",
        seed = 128694831,
    ),

    ### my
    ExpDesc(
        prompt = "the best place in Bayern",
        seed = 1,
        negative_prompt="",
    ),

    ### Liubov
    ExpDesc(
        prompt = "a photo of an astronaut riding a horse on mars",
        seed = 1,
        negative_prompt="",
    ),
    ExpDesc(
        prompt = "close-up photography of old man standing in the rain at night, in a street lit by lamps, leica 35mm summilux",
        seed = 1,
        negative_prompt="",
    ),

    # ExpDesc(
    #     prompt = "The spirit of a tamagotchi wandering in the city of Vienna",
    #     seed = 23,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "a beautiful pink unicorn, 8k",
    #     seed = 1,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "Super cute fluffy cat warrior in armor, photorealistic, 4K, ultra detailed, vray rendering, unreal engine",
    #     seed = 1,
    #     negative_prompt="",
    # ),
    # ExpDesc(
    #     prompt = "a train that is parked on tracks and has graffiti writing on it, with a mountain range in the background",
    #     seed = 1,
    #     negative_prompt="",
    # ),
]

def encode_prompt(prompt):
    return prompt.replace(' ', '_')[:20]

PROMPTS_MAP = {encode_prompt(desc.prompt): desc.prompt for desc in DESCS}