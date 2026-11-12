CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS rasters (
    raster_id  VARCHAR(16) PRIMARY KEY,          -- sha256(content)[:16] => идемпотентная загрузка
    meta       JSONB NOT NULL,                   -- crs, размер, разрешение, bounds
    status     VARCHAR(16) NOT NULL,             -- ingested | tiling | completed | failed
    extra      JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS landcover (
    id        BIGSERIAL PRIMARY KEY,
    raster_id VARCHAR(16) NOT NULL REFERENCES rasters(raster_id) ON DELETE CASCADE,
    class     VARCHAR(16) NOT NULL,
    area_m2   DOUBLE PRECISION NOT NULL,
    geom      geometry(Geometry) NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_landcover_geom ON landcover USING GIST (geom);      -- bbox / intersects
CREATE INDEX IF NOT EXISTS ix_landcover_raster_class ON landcover (raster_id, class);
CREATE INDEX IF NOT EXISTS ix_landcover_class_area ON landcover (class, area_m2 DESC);  -- «топ полигонов класса» без сортировки всей таблицы
