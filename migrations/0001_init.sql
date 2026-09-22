CREATE TABLE tenants (
  tenant_id    uuid PRIMARY KEY,
  display_name text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE properties (
  property_id  uuid PRIMARY KEY,
  tenant_id    uuid NOT NULL REFERENCES tenants,
  kind         text NOT NULL CHECK (kind IN ('stay','cafe')),
  display_name text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE rooms (
  room_id      uuid PRIMARY KEY,
  property_id  uuid NOT NULL REFERENCES properties,
  display_name text NOT NULL,
  slug         text NOT NULL,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (property_id, slug)
);

CREATE TABLE masters (
  master_id     uuid PRIMARY KEY,
  room_id       uuid NOT NULL REFERENCES rooms,
  variant       text NOT NULL DEFAULT 'default',   -- e.g. "4-chair", "2-chair"
  status        text NOT NULL CHECK (status IN ('PENDING','PROCESSING','ACTIVE','REJECTED','RETIRED')),
  s3_prefix     text NOT NULL,                     -- t/{..}/p/{..}/r/{..}/masters/{master_id}/
  quality_score real,
  reject_reason text,
  image_width   int,
  image_height  int,
  created_at    timestamptz NOT NULL DEFAULT now(),
  activated_at  timestamptz
);
-- at most one ACTIVE master per room per variant
CREATE UNIQUE INDEX one_active_master
  ON masters (room_id, variant) WHERE status = 'ACTIVE';

CREATE TABLE captures (
  capture_id     uuid PRIMARY KEY,                 -- UUIDv7, generated in app code
  room_id        uuid NOT NULL REFERENCES rooms,
  master_id      uuid REFERENCES masters,          -- master actually used for this comparison
  status         text NOT NULL CHECK (status IN ('PENDING','PROCESSING','REJECTED','COMPLETE','FAILED')),
  reject_reason  text,
  captured_at    timestamptz NOT NULL,             -- from device manifest
  s3_prefix      text NOT NULL,                    -- .../captures/{yyyy}/{mm}/{dd}/{capture_id}/
  ssim           real,
  brightness_delta real,
  findings       jsonb,                            -- compact delta summary (missing/drift/clutter)
  vlm_verdict    text,                             -- clean | minor_issue | needs_cleaning | cannot_assess
  checklist      jsonb,
  created_at     timestamptz NOT NULL DEFAULT now(),
  processed_at   timestamptz
);
CREATE INDEX captures_room_time ON captures (room_id, captured_at DESC);
CREATE INDEX captures_open ON captures (status) WHERE status IN ('PENDING','PROCESSING');