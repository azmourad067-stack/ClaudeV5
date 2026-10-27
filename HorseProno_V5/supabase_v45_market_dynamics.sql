-- Déjà appliqué au projet de travail. Fichier reproductible pour un autre projet.
alter table public.quinte_market_snapshots
  add column if not exists scheduled_start timestamptz,
  add column if not exists minutes_to_start numeric,
  add column if not exists odds_direct numeric,
  add column if not exists odds_reference numeric,
  add column if not exists trend_indicator text,
  add column if not exists trend_strength numeric,
  add column if not exists projected_odds_t2 numeric,
  add column if not exists predicted_log_move numeric,
  add column if not exists volatility_score numeric,
  add column if not exists dynamics_confidence numeric,
  add column if not exists dynamics_score numeric,
  add column if not exists model_rank integer,
  add column if not exists model_name text default 'HorseProno_V4_5';

create index if not exists idx_quinte_market_snapshots_race_capture
  on public.quinte_market_snapshots
  (race_date, meeting_number, race_number, captured_at);
