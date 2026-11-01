# Data sources

Licenses were read from the hosting pages, APIs or files on 2026-10-31.

| Dataset | Use in this project | Host | License as stated | Size |
|---|---|---|---|---|
| HPDmobile (6 homes, 152 labelled days) | long-term home occupancy (real labels) | figshare collection 5364449 | CC0 (figshare API, all 19 items) | 13.6 GB |
| DCASE 2016 Task 2 | isolated office events; synthetic test mixtures | archive.org | CC BY 3.0 (train/dev), CC BY 4.0 (test) | 0.45 GB |
| DCASE 2013 event detection (OL, OS) | isolated office events; real office recordings (OL test) | archive.org | CC BY 4.0 | 1.4 GB |
| LibriSpeech train-clean-100, dev-clean, test-clean | speech in training scenes; privacy attacks | openslr 12 | CC BY 4.0 | 7 GB |
| BUT Speech@FIT Reverb Database, RIR-only release 19_06 | room impulse responses and empty-room noise of 9 rooms | merlin.fit.vutbr.cz | project page: CC BY 4.0; read_me.txt: Apache 2.0 | 8.7 GB |
| DEMAND (OOFFICE, OMEETING, OHALLWAY, DLIVING, 16 kHz) | not used: the office recordings contain people | Zenodo 1227121 | Zenodo metadata: CC BY 4.0; DEMAND.pdf: CC BY-SA 3.0 | 0.3 GB |
| Whisper small.en (ASR for the content attack) | attack evaluation only | huggingface.co/openai/whisper-small.en | model card: Apache 2.0 | 1 GB |

## Notes

- HPDmobile audio is not raw. Each 10-s, 8 kHz clip was mean shifted, full-wave rectified and
  downsampled by a factor of 100 (80 values per second) to remove intelligible speech
  (Jacoby et al., Scientific Data 8, 280, 2021). The low lag-1 autocorrelation of the released
  values (0.29 to 0.44 on H1) is consistent with downsampling without low-pass filtering, so
  src/office_models.py:Envelope keeps every 100th sample.
- HPDmobile timestamps: the paper says the timestamp format is consistent across data types and
  gives no time zone. Around label transitions, the 1-minute loudness peaks within the 10 minutes
  after arrivals and before departures in all six homes, so audio and labels are aligned.
  The low loudness-occupancy AUC of H2, H4 and H6 at zero shift comes from quiet occupied
  periods (nights), not from a clock offset.
- DEMAND office recordings contain people (OOFFICE: three people using computers; OHALLWAY:
  people passing by; OMEETING: a discussion), so they cannot be used as empty-room noise.
- BUT ReverbDB: speaker setups with a second loudspeaker (background radio, $EnvSpk2 in
  mic_meta.txt; only L207 SpkID02) are excluded from the noise pool.
- Room split for office scenes: train = Q301 (office), L207 (office), L227 (stairs),
  Hotel Room112, Hotel ConferenceRoom2, E112 (lecture room); test = L212 (office),
  D105 (lecture room), C236 (meeting room).
- No public dataset found with raw office audio and occupancy ground truth. This is why our own
  recordings are planned (future work).
