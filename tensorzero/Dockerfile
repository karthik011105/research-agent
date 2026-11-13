# Pinned (tag + digest): `latest` could change gateway behaviour on any deploy without a code change.
# 2026.6.0 is the version `latest` pointed to while this config was developed and tested in production.
# To upgrade: pick a tag from https://hub.docker.com/r/tensorzero/gateway/tags, update both parts, run the tests.
FROM tensorzero/gateway:2026.6.0@sha256:c939db4f27e41aa3a37a87909430a0c8f898f9c952f402dcbdc17e1708597bcb
COPY tensorzero/tensorzero.toml /app/config/tensorzero.toml
COPY tensorzero/templates/ /app/config/templates/
CMD ["--config-file", "/app/config/tensorzero.toml"]
