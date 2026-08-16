"""
Rule-Based Clinical Insights Engine for Parkinson's Speech Therapy.
Implements evidence-based acoustic criteria modeled after LSVT LOUD
and clinical dysarthria assessments.
"""

from typing import Dict, Any, List, Optional
import numpy as np


class ClinicalVoiceEngine:
    """Evaluates acoustic measurements against Parkinsonian speech therapy thresholds."""

    TASK_SUSTAINED_VOWEL = "Sustained Vowel Phonation (/ah/)"
    TASK_READING = "Sentence Reading & Articulation"
    TASK_PARAGRAPH = "Paragraph Reading & Connected Speech"
    TASK_PITCH_GLIDE = "Pitch Glide & Dynamic Range"

    @classmethod
    def evaluate_session(
        cls,
        acoustics: Dict[str, Any],
        task_type: str = "Sustained Vowel Phonation (/ah/)",
        transcript_data: Optional[Dict[str, Any]] = None,
        target_db: float = 70.0
    ) -> Dict[str, Any]:
        """
        Generate strict rule-based clinical insights and actionable directives.
        
        Args:
            acoustics: Output dictionary from audio_processor.analyze_vocal_acoustics
            task_type: Type of speech therapy task performed
            transcript_data: Transcription details from Sarvam AI
            target_db: Target intensity baseline slider value
            
        Returns:
            Dict containing clinical score, findings, severity badges, and actionable therapy directives.
        """
        avg_db = acoustics.get("avg_db", 0.0)
        peak_db = acoustics.get("peak_db", 0.0)
        compliance_pct = acoustics.get("target_compliance_pct", 0.0)
        decay_slope = acoustics.get("loudness_decay_slope", 0.0)
        std_f0 = acoustics.get("std_f0", 0.0)
        mean_f0 = acoustics.get("mean_f0", 0.0)
        f0_range = acoustics.get("f0_range", 0.0)
        semitone_std = acoustics.get("semitone_std", 0.0)
        jitter_cv = acoustics.get("jitter_cv", 0.0)
        voicing_pct = acoustics.get("voicing_pct", 0.0)

        db_delta = avg_db - target_db

        insights: List[Dict[str, Any]] = []

        # =====================================================================
        # 1. HYPOPHONIA & INTENSITY ANALYSIS (LSVT LOUD CORE)
        # =====================================================================
        if db_delta < -7.0 or compliance_pct < 35.0:
            intensity_status = "Severe Hypophonia"
            intensity_color = "#E63946"  # Red
            intensity_badge = "CRITICAL DEFICIT"
            insights.append({
                "category": "Loudness & Vocal Effort",
                "status": "warning",
                "title": "Severe Vocal Hypophonia Detected",
                "finding": f"Average intensity was {avg_db:.1f} dB ({abs(db_delta):.1f} dB below the {target_db:.0f} dB target). Only {compliance_pct:.0f}% of voiced frames met target.",
                "rationale": "Parkinson's impairs sensory self-monitoring (patients underestimate how soft they sound) and reduces respiratory muscle drive.",
                "directives": [
                    "📣 **'Think LOUD!'** Calibrate your internal effort to feel like you are shouting; to listeners, it will sound like normal conversational volume.",
                    "🫁 **Diaphragmatic Breath Anchor**: Inhale deeply through your nose, expand lower ribs, and push sound out using your abdominal wall before speaking.",
                    "🎯 **Open Vocal Tract**: Drop your jaw and project your voice to a target across the room."
                ]
            })
        elif -7.0 <= db_delta < 0.0 or compliance_pct < 65.0:
            intensity_status = "Mild-to-Moderate Hypophonia"
            intensity_color = "#F4A261"  # Amber
            intensity_badge = "BELOW TARGET"
            insights.append({
                "category": "Loudness & Vocal Effort",
                "status": "caution",
                "title": "Vocal Intensity Below Baseline",
                "finding": f"Average intensity was {avg_db:.1f} dB ({abs(db_delta):.1f} dB below {target_db:.0f} dB target). Compliance: {compliance_pct:.0f}%.",
                "rationale": "Good initial breath support, but volume falls short of functional target for noisy environments.",
                "directives": [
                    "⚡ **Increase Effort by 20%**: Consciously amplify vocal energy on every syllable.",
                    "🗣️ **Sustain the Breath**: Do not allow breath pressure to dissipate mid-utterance."
                ]
            })
        else:
            intensity_status = "Optimal Projection"
            intensity_color = "#2A9D8F"  # Emerald Green
            intensity_badge = "TARGET ACHIEVED"
            insights.append({
                "category": "Loudness & Vocal Effort",
                "status": "success",
                "title": "Target Vocal Intensity Achieved!",
                "finding": f"Average intensity reached {avg_db:.1f} dB (+{db_delta:.1f} dB above baseline). Peak: {peak_db:.1f} dB. Compliance: {compliance_pct:.0f}%.",
                "rationale": "Patient successfully mobilized respiratory support to overcome Parkinsonian hypophonia.",
                "directives": [
                    "🌟 **Excellent Breath Control**: Maintain this level of abdominal engagement in daily conversation.",
                    "🔁 **Consistency Check**: Practice repeating this target volume across 5 consecutive trials."
                ]
            })

        # =====================================================================
        # 2. END-OF-PHRASE VOCAL DECAY (FATIGUE / RESPIRATORY DRIVE)
        # =====================================================================
        if decay_slope < -2.8:
            insights.append({
                "category": "Respiratory Endurance",
                "status": "warning",
                "title": "End-of-Utterance Volume Decay (Vocal Fade)",
                "finding": f"Intensity dropped at a rate of {abs(decay_slope):.1f} dB/second toward the end of the sample.",
                "rationale": "Respiratory muscle fatigue and premature exhalation cause trailing syllables to become inaudible.",
                "directives": [
                    "🫁 **Replenishment Breaths**: Take a quick top-up inhalation before long multi-word phrases.",
                    "🎯 **Punch the Final Word**: Exaggerate volume and articulation on the last 2 words of the sentence."
                ]
            })

        # =====================================================================
        # 3. PITCH DYNAMICS & MONOPITCH DETECTION (YIN ALGORITHM)
        # =====================================================================
        if task_type in (cls.TASK_READING, cls.TASK_PARAGRAPH):
            if std_f0 < 12.0 or semitone_std < 1.8:
                insights.append({
                    "category": "Pitch & Intonation (Prosody)",
                    "status": "caution",
                    "title": "Monopitch / Reduced Pitch Dynamics",
                    "finding": f"Pitch variation was restricted (SD: {std_f0:.1f} Hz, Semitone SD: {semitone_std:.1f} ST). Normal expressive speech exceeds 18 Hz.",
                    "rationale": "Laryngeal rigidity in Parkinson's limits pitch modulation, producing a flat, monotonous speech melody.",
                    "directives": [
                        "📈 **Pitch Exaggeration**: Practice lifting pitch on questions and emphasizing important nouns.",
                        "🎭 **Inflection Exercises**: Read phrases with exaggerated surprise or excitement to stretch cricothyroid vocal muscles."
                    ]
                })
            else:
                insights.append({
                    "category": "Pitch & Intonation (Prosody)",
                    "status": "success",
                    "title": "Healthy Pitch Dynamics & Prosody",
                    "finding": f"Dynamic pitch variation observed (SD: {std_f0:.1f} Hz, Semitone SD: {semitone_std:.1f} ST, Range: {f0_range:.1f} Hz).",
                    "rationale": "Patient exhibited flexible vocal cord modulation and expressive prosody.",
                    "directives": [
                        "👍 **Keep Dynamic Inflection**: Continue practicing expressive intonation during conversational tasks."
                    ]
                })

        elif task_type == cls.TASK_SUSTAINED_VOWEL:
            if jitter_cv > 8.5:
                insights.append({
                    "category": "Phonation Stability",
                    "status": "caution",
                    "title": "Vocal Tremor / Pitch Instability",
                    "finding": f"Pitch perturbation / instability detected during sustained vowel (Jitter CV: {jitter_cv:.1f}%).",
                    "rationale": "Subharmonic tremor in intrinsic laryngeal muscles is characteristic of Parkinsonian dysphonia.",
                    "directives": [
                        "🧘 **Relax Neck & Shoulder Muscles**: Avoid throat pinching; allow steady airflow from the diaphragm.",
                        "⏱️ **Focus on Steady Core**: Anchor the pitch at a comfortable, natural fundamental frequency."
                    ]
                })
            else:
                insights.append({
                    "category": "Phonation Stability",
                    "status": "success",
                    "title": "Stable Sustained Phonation",
                    "finding": f"Steady pitch holding with low tremor (Jitter CV: {jitter_cv:.1f}%, Pitch SD: {std_f0:.1f} Hz).",
                    "rationale": "Steady subglottic air pressure maintained across the 4-second phonation trial.",
                    "directives": [
                        "🎯 **Target Duration**: Aim to extend sustained /ah/ duration progressively to 8-10 seconds."
                    ]
                })

        elif task_type == cls.TASK_PITCH_GLIDE:
            if f0_range < 60.0:
                insights.append({
                    "category": "Pitch Flexibility",
                    "status": "warning",
                    "title": "Restricted Pitch Range",
                    "finding": f"Total pitch range reached was {f0_range:.1f} Hz (Goal: > 100 Hz).",
                    "rationale": "Stiffness in the vocal cord lengthening muscles restricts high-low pitch transitions.",
                    "directives": [
                        "🚨 **Siren Glide**: Slide your voice smoothly from lowest pitch up to high falsetto like a police siren."
                    ]
                })
            else:
                insights.append({
                    "category": "Pitch Flexibility",
                    "status": "success",
                    "title": "Excellent Pitch Range Flexibility",
                    "finding": f"Vocal pitch range expanded to {f0_range:.1f} Hz (Min: {acoustics.get('min_f0', 0):.0f} Hz -> Max: {acoustics.get('max_f0', 0):.0f} Hz).",
                    "rationale": "High vocal fold compliance and active laryngeal mobility.",
                    "directives": [
                        "🌟 **Maintain Range**: Perform pitch glides 3 times daily as a warm-up."
                    ]
                })

        # =====================================================================
        # 4. SARVAM AI TRANSCRIPTION & SPEECH RATE ANALYSIS
        # =====================================================================
        transcript_text = ""
        transcription_mode = "demo"
        duration_sec = acoustics.get("duration", 4.0)
        if transcript_data and transcript_data.get("transcript"):
            transcript_text = transcript_data.get("transcript", "").strip()
            transcription_mode = transcript_data.get("mode", "live")
            word_count = len(transcript_text.split())
            wpm = (word_count / max(duration_sec, 0.5)) * 60.0  # Dynamic words per minute over recorded duration

            if word_count > 0:
                if wpm > 190.0:
                    insights.append({
                        "category": "Speech Rate & Articulation",
                        "status": "caution",
                        "title": "Accelerated Speech Rate (Festination)",
                        "finding": f"Calculated speech rate: {wpm:.0f} WPM ({word_count} words in {duration_sec:.1f}s). Words appear rushed.",
                        "rationale": "Festinating speech in Parkinson's causes syllables to blur together as rate involuntarily increases.",
                        "directives": [
                            "🛑 **Pacing Strategy**: Insert conscious pauses between multi-word phrases.",
                            "🔤 **Over-Articulate Consonants**: Exaggerate 'p', 't', 'k', 's' sounds to maintain clarity."
                        ]
                    })
                elif wpm < 60.0 and task_type in (cls.TASK_READING, cls.TASK_PARAGRAPH):
                    insights.append({
                        "category": "Speech Rate & Articulation",
                        "status": "info",
                        "title": "Deliberate / Slow Pacing",
                        "finding": f"Speech rate: {wpm:.0f} WPM ({word_count} words in {duration_sec:.1f}s).",
                        "rationale": "Careful syllable timing observed. Maintain clear consonant articulation.",
                        "directives": [
                            "🗣️ **Maintain Smooth Flow**: Blend connected words while preserving target loudness."
                        ]
                    })

        # =====================================================================
        # 5. OVERALL COMPOSITE CLINICAL SCORE (0 - 100)
        # =====================================================================
        # Intensity Score: 40 pts
        intensity_score = np.clip((avg_db / target_db) * 40.0, 0.0, 40.0)
        
        # Compliance Score: 25 pts
        compliance_score = (compliance_pct / 100.0) * 25.0
        
        # Pitch / Stability Score: 20 pts
        if task_type == cls.TASK_SUSTAINED_VOWEL:
            stability_score = np.clip(20.0 - (jitter_cv * 1.5), 5.0, 20.0)
        elif task_type in (cls.TASK_READING, cls.TASK_PARAGRAPH):
            stability_score = np.clip((std_f0 / 20.0) * 20.0, 5.0, 20.0)
        else:
            stability_score = np.clip((f0_range / 100.0) * 20.0, 5.0, 20.0)

        # Voicing Continuity: 15 pts
        voicing_score = (voicing_pct / 100.0) * 15.0

        overall_score = float(np.clip(intensity_score + compliance_score + stability_score + voicing_score, 10.0, 100.0))

        if overall_score >= 80.0:
            grade = "A (Target Achieved - High Vocal Effort)"
            grade_color = "#2A9D8F"
        elif overall_score >= 60.0:
            grade = "B (Moderate Effort - Needs Louder Projection)"
            grade_color = "#F4A261"
        else:
            grade = "C (Hypophonic - Needs Diaphragmatic Push)"
            grade_color = "#E63946"

        return {
            "overall_score": round(overall_score, 1),
            "grade": grade,
            "grade_color": grade_color,
            "intensity_status": intensity_status,
            "intensity_color": intensity_color,
            "intensity_badge": intensity_badge,
            "insights": insights,
            "db_delta": round(db_delta, 1),
            "transcript": transcript_text,
            "transcription_mode": transcription_mode,
            "metrics": {
                "avg_db": round(avg_db, 1),
                "target_db": round(target_db, 1),
                "peak_db": round(peak_db, 1),
                "compliance_pct": round(compliance_pct, 1),
                "mean_f0": round(mean_f0, 1),
                "std_f0": round(std_f0, 1),
                "f0_range": round(f0_range, 1),
                "jitter_cv": round(jitter_cv, 1),
                "voicing_pct": round(voicing_pct, 1),
                "decay_slope": round(decay_slope, 2)
            }
        }
