SYMPTOM_SPECIALIZATION_MAP = {
    # Cardiology
    'chest pain': 'Cardiology',
    'heart palpitations': 'Cardiology',
    'shortness of breath': 'Cardiology',
    'high blood pressure': 'Cardiology',
    'low blood pressure': 'Cardiology',
    'irregular heartbeat': 'Cardiology',
    'swelling in legs': 'Cardiology',

    # Neurology
    'headache': 'Neurology',
    'dizziness': 'Neurology',
    'seizure': 'Neurology',
    'memory loss': 'Neurology',
    'migraine': 'Neurology',
    'numbness': 'Neurology',
    'tingling sensation': 'Neurology',
    'difficulty speaking': 'Neurology',
    'vision problems': 'Neurology',
    'tremors': 'Neurology',

    # Gastroenterology
    'stomach pain': 'Gastroenterology',
    'nausea': 'Gastroenterology',
    'vomiting': 'Gastroenterology',
    'diarrhea': 'Gastroenterology',
    'constipation': 'Gastroenterology',
    'bloating': 'Gastroenterology',
    'acid reflux': 'Gastroenterology',
    'blood in stool': 'Gastroenterology',
    'loss of appetite': 'Gastroenterology',

    # Dermatology
    'skin rash': 'Dermatology',
    'acne': 'Dermatology',
    'itching': 'Dermatology',
    'skin lesion': 'Dermatology',
    'dry skin': 'Dermatology',
    'eczema': 'Dermatology',
    'psoriasis': 'Dermatology',
    'hair loss': 'Dermatology',
    'nail discoloration': 'Dermatology',

    # Orthopedics
    'joint pain': 'Orthopedics',
    'back pain': 'Orthopedics',
    'fracture': 'Orthopedics',
    'muscle pain': 'Orthopedics',
    'knee pain': 'Orthopedics',
    'shoulder pain': 'Orthopedics',
    'swollen joints': 'Orthopedics',
    'difficulty walking': 'Orthopedics',

    # Pulmonology
    'chronic cough': 'Pulmonology',
    'wheezing': 'Pulmonology',
    'difficulty breathing': 'Pulmonology',
    'cough with blood': 'Pulmonology',
    'asthma symptoms': 'Pulmonology',

    # Endocrinology
    'fatigue': 'Endocrinology',
    'weight gain': 'Endocrinology',
    'weight loss': 'Endocrinology',
    'excessive thirst': 'Endocrinology',
    'frequent urination': 'Endocrinology',
    'sweating': 'Endocrinology',
    'heat intolerance': 'Endocrinology',

    # Psychiatry
    'depression': 'Psychiatry',
    'anxiety': 'Psychiatry',
    'insomnia': 'Psychiatry',
    'mood swings': 'Psychiatry',
    'hallucinations': 'Psychiatry',
    'panic attacks': 'Psychiatry',

    # Ophthalmology
    'blurred vision': 'Ophthalmology',
    'eye pain': 'Ophthalmology',
    'red eyes': 'Ophthalmology',
    'watery eyes': 'Ophthalmology',
    'double vision': 'Ophthalmology',

    # ENT
    'ear pain': 'ENT',
    'hearing loss': 'ENT',
    'sore throat': 'ENT',
    'nasal congestion': 'ENT',
    'runny nose': 'ENT',
    'loss of smell': 'ENT',

    # Urology
    'painful urination': 'Urology',
    'blood in urine': 'Urology',
    'urinary incontinence': 'Urology',
    'frequent urination': 'Urology',
    'difficulty urinating': 'Urology',

    # Gynecology
    'irregular periods': 'Gynecology',
    'pelvic pain': 'Gynecology',
    'vaginal discharge': 'Gynecology',
    'pain during intercourse': 'Gynecology',
    'pregnancy-related issues': 'Gynecology',

    # Pediatrics
    'crying excessively': 'Pediatrics',
    'fever in child': 'Pediatrics',
    'rashes in child': 'Pediatrics',
    'delayed milestones': 'Pediatrics',
}



# Symptom keywords for detection
SYMPTOM_KEYWORDS = [
    # General symptom verbs and descriptors
    'pain', 'ache', 'hurt', 'sore', 'tender', 'burning', 'stabbing', 'throbbing',
    'cramping', 'sharp pain', 'dull pain', 'pressure', 'tingling', 'numbness',
    'swelling', 'inflammation', 'discomfort', 'bruising', 'sensitivity',

    # Head & neurological
    'headache', 'migraine', 'dizziness', 'lightheadedness', 'vertigo',
    'seizure', 'tremors', 'memory loss', 'confusion', 'blurred vision',
    'double vision', 'difficulty speaking', 'loss of balance', 'fainting',

    # Digestive
    'nausea', 'vomiting', 'stomach pain', 'abdominal pain', 'bloating',
    'diarrhea', 'constipation', 'gas', 'indigestion', 'acid reflux',
    'loss of appetite', 'heartburn', 'cramps', 'blood in stool',

    # Respiratory
    'cough', 'sneeze', 'runny nose', 'nasal congestion', 'shortness of breath',
    'wheezing', 'chest pain', 'tight chest', 'difficulty breathing',
    'coughing up blood', 'phlegm', 'hoarseness',

    # Cardiovascular
    'heart palpitations', 'irregular heartbeat', 'fast heartbeat', 'slow heartbeat',
    'chest tightness', 'high blood pressure', 'low blood pressure',
    'swelling in legs', 'cold hands and feet',

    # Skin & allergy
    'rash', 'itching', 'dry skin', 'flaky skin', 'redness', 'hives',
    'skin peeling', 'eczema', 'acne', 'psoriasis', 'skin lesion',
    'discoloration', 'lumps', 'boils', 'blisters',

    # Musculoskeletal
    'joint pain', 'back pain', 'neck pain', 'muscle pain', 'stiffness',
    'limited mobility', 'cramps', 'spasms', 'bone pain', 'fracture',

    # Mental health
    'fatigue', 'tired', 'weakness', 'exhaustion', 'anxiety', 'depression',
    'stress', 'mood swings', 'irritability', 'insomnia', 'sleep problems',
    'hallucinations', 'panic attacks', 'feeling down', 'low energy',

    # Urology & genital
    'frequent urination', 'painful urination', 'burning while urinating',
    'blood in urine', 'urinary incontinence', 'vaginal discharge',
    'irregular periods', 'pelvic pain', 'pain during intercourse',

    # Eye & ENT
    'eye pain', 'red eyes', 'watery eyes', 'blurred vision', 'itchy eyes',
    'ear pain', 'hearing loss', 'ringing in ears', 'sore throat',
    'difficulty swallowing', 'loss of smell', 'loss of taste',

    # Infection & immune response
    'fever', 'chills', 'night sweats', 'body aches', 'swollen glands',

    # Common phrases for NLP
    'experiencing', 'feeling', 'having', 'suffering from', 'showing symptoms of',
    'dealing with', 'complaining of', 'symptoms', 'signs', 'issues',
    'problem', 'trouble', 'unwell', 'sick', 'not feeling well'
]