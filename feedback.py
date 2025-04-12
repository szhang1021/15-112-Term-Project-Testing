
import random
from textblob import TextBlob

def getPositiveFeedback(postureType):
    options = feedbackOptions.get(postureType, ["Keep going! You're improving."])
    feedback = random.choice(options)
    return feedback

feedbackOptions = {
    "Head Forward": [
        "You're doing a great job improving your head alignment!",
        "Keep it up — your neck posture is getting better each day.",
        "Way to go! Chin tucks are clearly making a difference.",
    ],
    "Rounded Shoulders": [
        "Nice work keeping those shoulders back!",
        "You're really opening up your chest — great progress.",
        "Strong posture today! Those shoulder rolls are paying off.",
    ]
}


