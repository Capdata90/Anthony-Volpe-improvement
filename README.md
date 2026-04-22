# Anthony Volpe: Performance Optimization & Predictive Analytics 

##  Executive Summary
This repository houses a multi-layered analytical project dedicated to optimizing the performance of **Anthony Volpe** (New York Yankees, SS). By leveraging **Statcast (MLBAM)** data and player-tracking insights inspired by my experience with **Hawk-Eye Innovations**, this project aims to identify marginal gains in Volpe's batting profile and defensive range to maximize his WAR (Wins Above Replacement). I beleive that Volpe is great player who can improve a lot if he is healthy and become the NYY SS for long term.

### Key Features
* **Strike Zone & Spray Analysis:** Granular contact data visualization.
* **Comparative Metrics:** Exit Velocity (EV) and Launch Angle (LA) breakdowns by pitch type.
* **Executive Business Case:** Real-time ROI simulation showing how a 2.0° LA increase can project a **$16M+ gain in Asset Value** (based on WAR metrics).

### Tech Stack
Python, Streamlit, Pandas, Plotly, GitHub.

---

##  Sub-Projects & Technical Ecosystem

### 1. Interactive Batting Performance Dashboard (Streamlit) 
# Anthony Volpe Offensive Optimization Dashboard
[Live App Link](https://anthony-volpe-improvement-project.streamlit.app/)
A high-level tool designed for scouts and coaches to visualize Volpe's offensive output in real-time.
* **Core Metrics:** Exit Velocity (EV) trends, Launch Angle (LA) distribution, and Zone-specific contact rates.
* **Key Feature:** Dynamic filtering by pitch velocity and type to identify "blind spots" in the strike zone.
* **Tech Stack:** `Python`, `Streamlit`, `Plotly`, `Seaborn`.

### 2. Plate Discipline & Swing Decision Analysis 
A deep-dive into Volpe’s decision-making process. This sub-project focuses on **O-Swing%** (swings at pitches outside the zone) vs. **Z-Swing%**.
* **Goal:** Identifying "Shadow Zone" tendencies to reduce strikeout rates and increase walk frequency.
* **Methodology:** Quadrant-based whiff analysis using Statcast coordinate data (`plate_x`, `plate_z`).

### 3. Defensive Efficiency & "First Step" Tracking
Drawing on my background with **Hawk-Eye**, this analysis examines Volpe's defensive range at Shortstop.
* **Focus:** Conversion rates on "50/50" balls and positioning efficiency relative to hitter spray charts.
* **Insight:** Correlation between sprint speed burst and successful putouts on high-leverage plays.

### 4. Pitch Sequence Prediction Model (Experimental)
A Machine Learning approach to anticipate the next pitch Volpe will face based on count, runners on base, and pitcher tendencies.
* **Model:** Random Forest / XGBoost Classifier.
* **Utility:** Providing the hitter with a "scouting report" probability for 2-strike counts.

---

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Capdata90/anthony-volpe-improvement.git](https://github.com/Capdata90/anthony-volpe-improvement.git.git)
