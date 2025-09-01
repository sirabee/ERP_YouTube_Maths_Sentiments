# SENTIMENT ANALYSIS ANNOTATION GUIDELINES

## YouTube Mathematics Education Comments - Manual Validation Study

### OVERVIEW

You are helping validate an AI system that analyses sentiment in YouTube comments about mathematics education. This is **blind annotation** - you cannot see the AI's predictions to ensure unbiased evaluation.

Your annotations will be compared directly with the AI model's outputs to measure system performance.

---

## ANNOTATION FIELDS EXPLAINED

### 1. **manual_sentiment**

**What to annotate:** The overall emotional tone of the comment
**Required values:** `positive`, `negative`, or `neutral`

**Definitions:**

- **positive** - Expresses appreciation, gratitude, enthusiasm, satisfaction, or positive emotions toward mathematics/teaching
- **negative** - Expresses frustration, confusion, criticism, dissatisfaction, or negative emotions
- **neutral** - Factual, informational, or balanced without clear emotional lean

**Examples:**

- **Positive:** "Thank you so much! This really helped me understand calculus!"
- **Positive:** "Amazing explanation, you're the best math teacher!"
- **Negative:** "I hate math, it's so confusing and makes no sense"
- **Negative:** "This video is terrible, doesn't explain anything properly"
- **Neutral:** "The video covers quadratic equations in 15 minutes"
- **Neutral:** "Can you do a video on derivatives next?"

**Decision Guidelines:**

- Focus on the **overall tone**, not individual words
- Consider **educational context** - learning frustration may still be positive if constructive
- When mixed sentiment appears, choose the **dominant** emotional tone

---

### 2. **manual_learning_journey**

**What to identify:** Comments showing emotional progression from negative to positive within the same comment
**Required values:** `yes` or `no`

**Definition:**
A learning journey is a comment that explicitly shows transition from confusion/frustration → understanding/gratitude **within the same comment**.

**Learning Journey Examples (mark `yes`):**

- "I was so lost with fractions but after watching this I finally get it! Thank you!"
- "At first I thought this was impossible but now I see the pattern. Amazing!"
- "I used to hate algebra until I found your channel. Now I actually enjoy it!"
- "Struggled with this for hours, then watched your video and everything clicked!"

**NOT Learning Journeys (mark `no`):**

- **Only positive:** "Great explanation, very helpful"
- **Only negative:** "This is too hard for me"
- **Neutral progression:** "First we do X, then we do Y"
- **General improvement:** "Your videos have helped me improve" (no specific frustration→understanding)

**Key Requirements:**

- Must show **explicit progression** within the comment
- Must include **both negative and positive** emotional states
- Must be about **learning/understanding**, not just general improvement

---

### 3. **manual_has_transition**

**What to identify:** Comments that show any change in sentiment/topic within the comment
**Required values:** `yes` or `no`

**Definition:**
A transition occurs when the comment shifts between different sentiments, topics, or focuses within the same comment.

**Transition Examples (mark `yes`):**

- "The beginning was confusing but the end was clear" (negative → positive)
- "I love algebra but hate geometry" (positive → negative)
- "Thanks for the help. Can you do calculus next?" (gratitude → request)
- "This is hard. My teacher doesn't explain it this way." (difficulty → comparison)

**No Transition Examples (mark `no`):**

- "Thank you so much for this amazing explanation!" (consistently positive)
- "I don't understand any of this at all" (consistently negative)
- "Please do more videos on this topic" (consistently neutral/request)

**Note:** All learning journeys have transitions, but not all transitions are learning journeys.

---

### 4. **manual_confidence_0_to_1**

**What to rate:** Your confidence in your sentiment judgment
**Required values:** Decimal between `0.0` and `1.0`

**Scale:**

- **0.0-0.2** - Very uncertain, could easily be wrong
- **0.3-0.4** - Somewhat uncertain, multiple interpretations possible
- **0.5-0.6** - Moderately confident, some ambiguity remains
- **0.7-0.8** - Quite confident, sentiment is fairly clear
- **0.9-1.0** - Completely confident, sentiment is obvious

**Examples:**

- **0.9:** "Thank you so much! This saved my exam!" (clearly positive)
- **0.5:** "Well, I guess this helps a bit maybe..." (ambiguous)
- **0.2:** "Nice work 🙄" (could be sincere or sarcastic)

---

### 5. **annotator_notes**

**What to record:** Explain your reasoning, especially for difficult cases
**Format:** Free text

**When to use:**

- Sentiment is ambiguous or unclear
- Cultural/language factors affect interpretation
- Sarcasm or irony detected
- Multiple topics with different sentiments
- Any factor that influenced your decision

**Examples:**

- "Sarcasm detected, marked as negative despite positive words"
- "Mixed sentiment - frustration with topic but gratitude to teacher, chose positive for overall tone"
- "Non-native English, interpreted based on context"

---

### 6. **annotation_difficulty**

**What to rate:** How difficult this comment was to annotate
**Required values:** `easy`, `medium`, or `hard`

**Guidelines:**

- **easy** - Immediately obvious, no hesitation
- **medium** - Required some thought or consideration
- **hard** - Difficult decision, unsure of final choice

---

## ANNOTATION PRINCIPLES

### **Consistency Guidelines:**

1. **Focus on emotional tone** over technical content
2. **Consider educational context** - constructive criticism vs. destructive negativity
3. **Learning journeys require explicit progression** language
4. **Document uncertainty** in annotator_notes
5. **Take breaks** to maintain consistency

### **Common Challenges:**

**Sarcasm/Irony:**

- Mark difficulty as "hard"
- Explain detection in notes
- Focus on intended emotional message

**Mixed Sentiment:**

- Choose dominant emotional tone
- Note the complexity in annotator_notes
- Consider overall message impact

**Cultural/Language Variations:**

- Consider context and intent
- Note language factors in annotator_notes
- Focus on emotional content over grammar

**Educational vs. Personal Sentiment:**

- Distinguish between frustration with math vs. gratitude to teacher
- Consider the primary target of sentiment
- Note complexity in annotator_notes

### **Quality Assurance:**

- Annotate in focused sessions
- Review difficult cases
- Use annotator_notes extensively
- Maintain consistent standards throughout

---

## ANNOTATION WORKFLOW

1. **Read the entire comment** before making any judgments
2. **Identify overall sentiment** (positive/negative/neutral)
3. **Check for learning journey** (explicit negative→positive progression)
4. **Look for any transitions** (sentiment or topic changes)
5. **Rate your confidence** in the sentiment judgment
6. **Record difficulty level** and any explanatory notes
7. **Review before moving to next comment**
