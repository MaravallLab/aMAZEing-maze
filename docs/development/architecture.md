# Architecture

## The shape of the system

```mermaid
flowchart TB
    subgraph app["Graphical application (amazeing.app)"]
        form[Config form<br/>one panel per mode]
        wave[Stimulus preview<br/>waveform, spectrum, spectrogram]
        panel[Process panel<br/>console and controls]
    end

    subgraph pkg["Package (amazeing.auditory)"]
        cfg[ExperimentConfig<br/>every session setting]
        fac[ExperimentFactory<br/>builds the trial table]
        aud[Audio<br/>synthesis and speaker compensation]
        vis[ROIMonitor<br/>arm occupancy]
        hw[Camera and Arduino]
        dm[DataManager<br/>folders, logs, manifest]
        an[SessionAnalyzer<br/>figures]
    end

    yaml[(session YAML)]
    data[(session folder<br/>CSVs, video, manifest)]

    form -->|writes| yaml
    yaml -->|read by| cfg
    panel -->|launches| cfg
    cfg --> fac --> aud
    cfg --> vis
    cfg --> hw
    fac --> loop
    loop[Session loop<br/>main.py]
    vis --> loop
    hw --> loop
    loop --> dm --> data
    data --> an
    cfg -.->|same objects| wave
```

The application never runs an experiment in its own process. It writes a configuration file and launches the same command-line entry point a terminal user would, then shows that process's output. This is why a session started from the window and one started from a terminal are identical, and why the interface cannot introduce a behaviour the scripts do not have.

## The session loop

```mermaid
sequenceDiagram
    participant C as Camera
    participant V as ROIMonitor
    participant M as Session loop
    participant A as Audio
    participant D as DataManager

    Note over M: Calibration: measure each arm's<br/>empty brightness (maze must be empty)
    loop every frame
        C->>M: frame
        M->>V: thresholded frame
        V-->>M: arms entered this frame
        alt an arm was entered
            M->>A: play that arm's stimulus
            M->>D: start the visit timer
        end
        alt all arms empty
            M->>A: stop
        end
        alt an arm was left
            M->>D: log the visit
        end
    end
    Note over M,D: At each block boundary, visits still<br/>open are closed and logged
```

## The nine-block cycle

Every mode uses the same structure. Odd positions are silent, even positions are active, and the stimulus-to-arm mapping is reshuffled at the start of each active block.

```mermaid
flowchart LR
    B1[1<br/>silent] --> B2[2<br/>active] --> B3[3<br/>silent] --> B4[4<br/>active]
    B4 --> B5[5<br/>silent] --> B6[6<br/>active] --> B7[7<br/>silent] --> B8[8<br/>active] --> B9[9<br/>silent]

    style B2 fill:#1565C0,color:#fff
    style B4 fill:#1565C0,color:#fff
    style B6 fill:#1565C0,color:#fff
    style B8 fill:#1565C0,color:#fff
```

The reshuffle is the load-bearing part of the design. If an animal returns to the same physical arm regardless of what plays there, that is a place preference. If it follows a stimulus as the stimulus moves between arms, that is a sound preference. Without the reshuffle the two are indistinguishable.

Because each active block needs a mapping that has not been used yet, and there are only as many mappings as there are orderings of the arms, a maze with fewer than three arms cannot supply four distinct mappings. The search gives up after a bounded number of attempts and reuses one rather than looping forever.

## Data flow through an experiment

```mermaid
flowchart LR
    subgraph session["During the session"]
        s1[trials CSV<br/>per block and arm]
        s2[detailed visits CSV<br/>per visit]
        s3[maze entries CSV]
        s4[video]
        s5[manifest]
    end

    subgraph post["Afterwards"]
        p1[Per-session figures]
        p2[Cross-session summary<br/>figures and tables]
        p3[Pose estimation<br/>crop, infer, filter]
        p4[Statistics and models]
    end

    s1 --> p1 --> p2 --> p4
    s2 --> p1
    s3 --> p1
    s4 --> p3 --> p4
    s5 -.->|units and settings| p2
```

## Package layout

| Module | Responsibility |
|---|---|
| `amazeing.auditory.config` | `ExperimentConfig`: every setting, the block schedule, the per-mode stimulus parameters |
| `amazeing.auditory.session_config` | Reading and writing that config as YAML |
| `amazeing.auditory.experiments` | `ExperimentFactory`: turns a config into a trial table plus waveforms |
| `amazeing.auditory.audio` | Synthesis, speaker compensation, playback |
| `amazeing.auditory.vision` | `ROIMonitor`: occupancy with debouncing |
| `amazeing.auditory.hardware` | Camera and Arduino |
| `amazeing.auditory.data_manager` | Session folders, visit logs, the manifest and its column units |
| `amazeing.auditory.analysis` | Per-session figures, and the colour palette shared with the application |
| `amazeing.auditory.summary_analysis` | Cross-session figures and tables |
| `amazeing.auditory.grammar_stimuli` | Markov grammars, tone synthesis, training-day playback |
| `amazeing.simplermaze` | Tactile paradigm |
| `amazeing.app` | The graphical interface |

## Design rules

**The application is a layer, not a fork.** Anything it can do, the command line can do. If a feature would only work in the window, it belongs in the package instead.

**The configuration file is the contract.** The form, the YAML file and the dataclass describe the same thing, and a round trip through any of them changes nothing. Unknown fields are rejected rather than ignored, so a typo cannot silently leave a default in place.

**Defaults reproduce the published protocols.** The per-mode stimulus values were literals in the code until they became configurable. Their defaults are pinned by tests so they cannot drift away from what was actually run.

**Units live in the data.** Every session writes a manifest describing its own columns, because the same column name meant milliseconds in the first version of this software and seconds in the second.
