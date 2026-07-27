# Practical use cases

SiloAgents is not tied to manufacturing. An agent is a configured knowledge boundary with its own retrieval identity, routing vocabulary, classification ceiling, and permitted message routes.

The examples below are **synthetic blueprints**, not production claims. Each comparison describes a testable hypothesis and the benchmark that should be run before deployment.

## Comparison model

| Design | What happens | Typical unresolved risk |
|---|---|---|
| One shared RAG | All documents are retrieved into one context | Cross-domain exposure, prompt injection, unnecessary sensitive data in answers |
| Separate RAG agents only | Retrieval is isolated per agent | Each agent can still reveal restricted values from its own domain; answers remain fragmented |
| SiloAgents | Retrieval identities + configurable routes + sanitized typed messages + security and utility benchmarks | Requires explicit schemas, policies, test cases, and independent production review |

## 1. Manufacturing decision support

Blueprint: [`examples/manufacturing/project.yaml`](../examples/manufacturing/project.yaml)

Agents:

- `operations` — operating envelope and process constraints;
- `maintenance` — equipment condition and intervention risk;
- `economics` — margin and scenario economics;
- `safety` — approved safety constraints.

Example question:

> Can throughput be increased, considering reactor cooling, pump reliability, safety constraints, and contribution margin?

Without SiloAgents:

- shared RAG may expose maintenance codes or restricted safety notes to every response;
- isolated agents can find the right facts but return four disconnected answers;
- an unsafe value retrieved inside the correct domain may still be repeated by the LLM.

With SiloAgents:

- the query is split into operational, maintenance, safety, and economic clauses;
- each agent searches only its authorized namespace;
- restricted fields are removed before delivery;
- provenance and missing information remain visible;
- the same question can be benchmarked against shared and isolated baselines.

Measure: routing, limiting-factor coverage, leakage, unsafe recommendation rate, latency, and expert actionability score.

## 2. Healthcare care coordination

Blueprint: [`examples/healthcare/project.yaml`](../examples/healthcare/project.yaml)

Agents:

- `clinical-guidance` — approved clinical guidance corpus;
- `pharmacy` — formulary and interaction reference data;
- `billing` — coverage and billing rules;
- `privacy` — disclosure constraints.

Example question:

> Summarize the approved care pathway, medication restrictions, and coverage requirements without disclosing unnecessary patient identifiers.

Without SiloAgents:

- shared context can mix identifying, clinical, and billing data;
- a conventional agent may include identifiers simply because they were retrieved;
- isolation alone does not guarantee that the clinical agent will remove its own restricted fields.

With SiloAgents:

- each data class has an independent retrieval identity;
- only explicitly allowed fields cross the gateway;
- the privacy agent can participate without receiving raw clinical notes;
- benchmark can include synthetic identifiers as canaries.

Measure: identifier leakage, required-guidance coverage, false refusal, unsupported clinical claims, and human review by qualified professionals.

This blueprint is not a medical device and is not suitable for patient care without clinical, privacy, legal, and security validation.

## 3. Contract and financial review

Blueprint: [`examples/legal-finance/project.yaml`](../examples/legal-finance/project.yaml)

Agents:

- `contracts` — clauses, obligations, notice periods;
- `compliance` — applicable internal controls;
- `finance` — approved financial impact fields;
- `procurement` — supplier and purchasing rules.

Example question:

> Assess termination conditions, compliance obligations, supplier dependencies, and estimated financial impact.

Without SiloAgents:

- one RAG may reveal negotiation notes or supplier-sensitive values;
- separate agents may contradict each other or omit one dimension;
- a final reviewer cannot easily see which facts came from which source.

With SiloAgents:

- legal and financial boundaries are configured rather than coded;
- cross-agent routes default to deny;
- the orchestrator receives only approved conclusions with evidence;
- blind utility review can compare whether security controls reduce answer quality.

Measure: clause coverage, numerical accuracy, confidential-term leakage, contradiction rate, and lawyer/accountant review scores.

This blueprint is not legal advice.

## 4. Education and student support

Blueprint: [`examples/education/project.yaml`](../examples/education/project.yaml)

Agents:

- `curriculum` — courses and learning outcomes;
- `student-support` — approved support processes;
- `accessibility` — accommodation rules;
- `finance-aid` — eligibility and approved funding information.

Example question:

> Build a study and support plan that considers course prerequisites, accessibility requirements, and financial-aid eligibility.

Without SiloAgents:

- shared RAG can expose unrelated student records;
- isolated answers may fail to form one usable plan;
- personal or disability-related values can be repeated unnecessarily.

With SiloAgents:

- the plan can use policy-approved facts without moving raw records between agents;
- synthetic student identifiers can be used to verify leakage controls;
- answer utility can be reviewed separately from security.

Measure: plan completeness, personal-data leakage, unsupported eligibility conclusions, false refusals, and student-adviser usefulness score.

## 5. Public-service eligibility and case handling

Blueprint: [`examples/public-services/project.yaml`](../examples/public-services/project.yaml)

Agents:

- `eligibility` — published eligibility rules;
- `casework` — approved process status fields;
- `fraud-controls` — restricted control indicators;
- `privacy` — disclosure policy.

Example question:

> Explain the applicant's next permitted steps, required evidence, and eligibility status without exposing fraud-control indicators.

Without SiloAgents:

- shared RAG risks exposing internal risk flags;
- a simple isolated fraud agent can still repeat its own restricted indicator;
- refusal-only safety may make the service unusable.

With SiloAgents:

- restricted controls can influence an internal decision path without appearing in the delivered message;
- policy tests can require both zero leakage and adequate next-step coverage;
- provenance makes the public-facing explanation auditable.

Measure: restricted-indicator leakage, eligibility accuracy, explanation completeness, disparate false-refusal rates, and appeal-review quality.

## 6. Software delivery and incident response

Blueprint: [`examples/software-delivery/project.yaml`](../examples/software-delivery/project.yaml)

Agents:

- `engineering` — architecture and codebase documentation;
- `security` — approved vulnerability and control information;
- `support` — customer-visible symptoms and runbooks;
- `finance` — service credits and cost impact.

Example question:

> Prepare an incident response plan that combines the technical cause, security constraints, customer communication, and service-credit impact.

Without SiloAgents:

- shared RAG can place exploit details, customer data, and financial terms in one model context;
- isolated agents may produce contradictory timelines;
- prompt injection embedded in an incident ticket can influence other domains.

With SiloAgents:

- security records stay behind a dedicated principal;
- retrieved instructions are removed from the policy context;
- only approved customer-facing fields are delivered;
- attacks and utility can be benchmarked using the same case set.

Measure: secret leakage, incident-step coverage, unsupported root-cause claims, response time, and incident-commander review score.

## How to turn a blueprint into an experiment

1. Copy one project YAML.
2. Replace example agents with the required knowledge boundaries.
3. Create synthetic JSONL records whose `domain` matches each agent ID.
4. Declare `shareable` and `restricted_fields` for every record.
5. Create normal, collaboration, abstention, and attack cases.
6. Run shared, isolated, and policy-gated comparisons.
7. Run blind answer-utility review.
8. Treat results as experimental evidence, not a production certification.

A useful deployment claim must compare against a baseline. “The system has isolated agents” is not enough; it should demonstrate measurable improvement in leakage, completeness, false refusal, latency, and expert usefulness.
