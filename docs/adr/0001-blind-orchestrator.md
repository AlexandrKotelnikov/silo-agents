# ADR 0001: Use a blind orchestrator

## Status

Accepted.

## Context

A central orchestrator that can query every domain store becomes a privileged aggregation point and a prompt-injection target.

## Decision

The orchestrator receives only capability metadata, relevance ACKs, and policy-approved typed conclusions. It has no direct retriever or document credentials.

## Consequences

Positive: smaller blast radius, clearer provenance, measurable information flow, and easier leakage testing.

Negative: routing can be less accurate; cross-domain synthesis requires more exchanges; relevance ACKs must also be minimized.
