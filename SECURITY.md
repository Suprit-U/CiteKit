# Security Policy

## Supported versions

The latest released version receives security fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Email the maintainer with:

- the affected version,
- a description of the issue and its impact,
- a minimal reproduction if you have one.

You will get an acknowledgement within a few days and a fix or an explanation of why
the report is not considered a vulnerability.

## Scope

CiteKit is a local library. It has no server component by default, no authentication,
and no telemetry. When you run the optional FastAPI server, it binds where you tell it
to and has no access control of its own — do not expose it to an untrusted network
without putting your own authentication in front of it.
