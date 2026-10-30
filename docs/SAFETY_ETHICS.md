# Safety, ethics, limitations

- **Decision support, not decision maker.** Official evacuation orders come from authorities.
  The API returns a disclaimer; the UI must too.
- **Failure modes**: under-predicting peaks (deadly) vs over-predicting (alert fatigue). Tune for
  recall on dangerous depths; report calibrated probabilities.
- **Out-of-distribution events**: embankment breaches, cyclone surge, dam release — simulate them
  and stress-test; fall back to conservative plans when inputs look unfamiliar.
- **Equity**: routes and shelters must serve elderly, disabled, women/children, low-income and
  informal settlements; evaluate success rate per group, not only on average.
- **Privacy**: aggregate/anonymise location data; follow local law and humanitarian data standards.
- **Validation**: never deploy on simulator-only results; require hindcast validation on real events
  with local hydrologists.
- **Human factors**: multilingual (Bangla/English), low-bandwidth, SMS/voice alerts, clear maps.
