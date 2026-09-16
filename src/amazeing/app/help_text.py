"""Explanations shown behind the "What does this do?" toggles.

Each entry describes what the section controls, why it matters for the
behaviour being measured, and the practical consequence of getting it wrong.
Keys match the group-box keys in ``config_form.ConfigForm.boxes``.
"""

SECTION_HELP = {
    "experiment": (
        "A session always runs the same nine-block cycle: silent, active, silent, "
        "active, and so on. Stimuli play only during the active blocks, and the "
        "mapping of stimulus to arm is reshuffled at the start of each active "
        "block. That shuffle is the point of the design: if an animal keeps "
        "returning to the same physical arm regardless of what plays there, that "
        "is a place preference, not a sound preference, and the reshuffle lets you "
        "separate the two afterwards.<br><br>"
        "<b>Number of arm ROIs</b> must match the arms you draw on the camera "
        "image. <b>Day label</b> only groups sessions into folders, which the "
        "cross-session analysis then reads as your experimental days."
    ),
    "devices": (
        "<b>Sample rate</b> has to be at least twice the highest frequency you "
        "intend to play. Mouse-audible stimuli commonly run to 25 kHz or beyond, "
        "so 192 kHz is the usual choice and an ordinary sound card will not "
        "deliver it; an ultrasonic interface and speaker are required.<br><br>"
        "<b>Audio output device</b> is the index the sound library reports, not "
        "the Windows default. If you hear nothing, this is the first thing to "
        "check.<br><br>"
        "<b>TTL pulses</b> are only needed when something else has to be aligned "
        "to the sound, such as fibre photometry. The pulse goes high when a sound "
        "starts and low when it ends or the animal leaves the arm."
    ),
    "detection": (
        "Detection is by infrared silhouette, not by tracking. The camera sees a "
        "bright floor; the frame is thresholded so the floor is white and the "
        "animal is black, and an arm counts as occupied when the white pixels "
        "inside its rectangle drop below a fraction of the empty-maze baseline "
        "measured at the start of the session.<br><br>"
        "<b>Binary threshold</b> is that black-and-white cut in pixel values, and "
        "depends on your illumination. <b>Detection sensitivity</b> is the "
        "fraction: 0.6 means an arm is occupied once it is 40 per cent darker than "
        "when empty. Too high and shadows trigger entries; too low and a small "
        "animal at the edge of an arm is missed. Watch the binary view during a "
        "test session and adjust until entries match what you see.<br><br>"
        "Keep the maze empty during the calibration frames at session start, or "
        "every baseline will be wrong."
    ),
    "sound": (
        "<b>Sound duration</b> is how long one presentation lasts. Playback starts "
        "when the animal enters an arm and stops when it leaves, so this is an "
        "upper bound rather than a fixed length.<br><br>"
        "<b>Onset ramp</b> fades the sound in. Without it the abrupt start "
        "produces a broadband click that the animal can hear even when the tone "
        "itself is inaudible, which would confound any frequency comparison.<br><br>"
        "<b>Speaker calibration</b> compensates for the fact that no speaker is "
        "equally loud at every frequency. The file lists attenuation against "
        "frequency, and each tone is boosted by the attenuation at its own "
        "frequency so the set is equally loud. Without it, a preference for one "
        "tone may only mean that tone was louder. Edit the curve in the Speaker "
        "calibration tab."
    ),
    "simple_smooth": (
        "One pure tone per arm, the simplest frequency-preference design. Give one "
        "frequency per arm; if you give fewer, the list is recycled, which means "
        "two arms would play the same tone.<br><br>"
        "Stay within the range your speaker can actually produce, and remember "
        "that perceived loudness still depends on the calibration curve."
    ),
    "simple_intervals": (
        "Each arm plays two tones together: a fixed tonal centre and one interval "
        "above it. Because the lower tone is shared, the arms differ only in the "
        "frequency ratio between the pair, which is what makes an interval "
        "consonant or dissonant.<br><br>"
        "Intervals use just intonation, so a perfect fifth is exactly 3:2. A "
        "unison arm and a silent arm are added automatically, so tick two fewer "
        "intervals than you have arms. The order you tick them in is the order "
        "they are assigned to arms in the first active block."
    ),
    "tem": (
        "Amplitude modulation varies the loudness of a carrier tone over time "
        "without changing its pitch. This separates the temporal envelope of a "
        "sound from its spectral content, so you can ask whether an animal "
        "responds to rhythm rather than frequency.<br><br>"
        "<b>Smooth</b> arms play an unmodulated carrier. <b>Constant AM</b> arms "
        "modulate at one steady rate. <b>Complex AM</b> arms switch between "
        "several rates in 200 ms segments, giving a less predictable envelope. "
        "<b>Depth</b> sets how deep the modulation goes, from none at 0 to full "
        "at 1.<br><br>"
        "Arms are filled in order: controls first, then smooth, constant, complex. "
        "The totals must add up to the number of arms."
    ),
    "complex_intervals": (
        "The consonant against dissonant contrast, run as a multi-day protocol. "
        "Each day presents a different published set of intervals so that a "
        "preference can be tested against several pairs rather than one.<br><br>"
        "Keep the preset ticked to run a day exactly as published. Untick it to "
        "build your own set; the day name then only labels the output folder. The "
        "optional unison arms give a same-frequency reference: one smooth, one "
        "amplitude-modulated, which controls for roughness independently of the "
        "interval ratio."
    ),
    "sequences": (
        "Patterns of tones over time rather than single sounds, which lets you ask "
        "about structure and repetition. Each letter is a tone and 'o' is a silent "
        "slot, so AoAo is a tone alternating with silence while ABAB alternates "
        "two tones.<br><br>"
        "Give every letter a frequency in the tone map. The special names silence, "
        "vocalisation and random behave as their names suggest; random draws from "
        "the mapped tones each time. Leaving the tone map empty falls back to the "
        "original console prompts, which the interface cannot answer for you."
    ),
    "vocalisation": (
        "Every .wav file in the chosen folder becomes one arm, in the order the "
        "folder lists them, so the number of files should match the number of "
        "arms. Conspecific calls are biologically meaningful in a way synthetic "
        "tones are not, which is why they often behave differently from every "
        "other stimulus in these experiments.<br><br>"
        "Files are resampled to the session sample rate. Recordings made at a "
        "lower rate will not gain ultrasonic content by being resampled."
    ),
    "grammar": (
        "Two artificial grammars over six tones, each a set of transition "
        "probabilities rather than a fixed melody. During training an animal hears "
        "one grammar in its enriched cage and the other in its standard cage, so "
        "each grammar becomes associated with an environment. The maze then tests "
        "whether that association shows up as arm preference.<br><br>"
        "<b>Silent baseline</b> is the habituation day: one continuous block, no "
        "audio, establishing which arms the animal likes before any sound is "
        "involved. <b>Test</b> runs the full nine-block cycle.<br><br>"
        "<b>Enriched grammar</b> must match what this individual animal heard "
        "during training, and it differs between animals by design. Setting it "
        "wrong inverts that animal's data. Set a <b>seed</b> only if you need the "
        "same melodies twice."
    ),
    "custom": (
        "Your own stimulus on each arm, without using one of the published "
        "designs. Arms with no row play nothing, which is how you add a silent "
        "control.<br><br>"
        "<b>tone</b> is a pure tone and needs a frequency. <b>am_tone</b> adds "
        "amplitude modulation and takes a rate and a depth. <b>wav</b> plays a "
        "file, resampled to the session rate. The label is free text and appears "
        "in the trials table and the figures, so name things the way you want to "
        "read them later."
    ),
    "voc_files": (
        "The folder supplies one arm per file in vocalisation mode. The control "
        "file is the single recording used on the vocalisation arm of the mixed "
        "designs.<br><br>"
        "A missing file is not loud: the loader returns silence, so the session "
        "would run with a silent arm where a recording was meant to be, and "
        "nothing in the data would say so. The Sound files line under Experiment "
        "therefore checks, and a session will not start until every recording "
        "the mode needs has been found.<br><br>"
        "If you do not want a vocalisation arm at all, untick it in the mode's "
        "control arms rather than leaving the file empty."
    ),
}
