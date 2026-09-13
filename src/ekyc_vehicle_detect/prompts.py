"""Zero-shot class prompts for CLIP.

Each class is described with several phrasings ("prompt ensembling"), which
CLIP zero-shot classification uses to reduce sensitivity to exact wording.
The two classes are deliberately framed as opposites of the same question
("is the camera inside a car right now?") rather than generic scene labels,
since eKYC frames are dominated by a face/torso and the vehicle cues are
peripheral (dashboard, windows, seatbelt, door trim, headrests).
"""

VEHICLE_PROMPTS = [
    "a photo taken inside a car",
    "a selfie video filmed inside a moving vehicle",
    "a person sitting in the driver seat of a car",
    "a person sitting in the passenger seat of a car",
    "the interior of a car showing a dashboard and steering wheel",
    "a car window with the road or street visible outside",
    "a person wearing a seatbelt inside a vehicle",
    "the interior of a car with a headrest and door panel visible",
    "a person recording a video while inside a taxi or rideshare car",
    "a car's rearview mirror or sun visor in the background",
]

NON_VEHICLE_PROMPTS = [
    "a photo taken indoors in a room, not in a vehicle",
    "a selfie video filmed in a bedroom or living room",
    "a person sitting at a desk or table indoors",
    "a person standing outdoors on a street or sidewalk, not inside a car",
    "a person in an office or workplace setting",
    "a plain indoor wall or curtain in the background",
    "a person in an outdoor natural setting such as a park or garden",
    "a video call style selfie against a home background",
    "a person in a public place such as a shop, bank, or hallway",
    "a close-up face photo with no vehicle parts visible in the background",
]

PROMPT_TEMPLATES = [
    "{}",
    "a video frame showing {}",
    "an image that shows {}",
]

CLASS_VEHICLE = "in_vehicle"
CLASS_NON_VEHICLE = "not_in_vehicle"
