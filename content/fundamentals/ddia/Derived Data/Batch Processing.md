---
title: Batch Processing
---
So far we’ve talked about requests/queries and responses/results (e.g. HTTP-REST-based APIs), but this isn’t the only way to build data systems

3 types of systems
- Services (online)
	- Service- waits for request and then handles with response as quickly as possible
	- Metrics
		- Reponse time (primary)
		- Availability (also important)
- Batch Processing Systems (offline)
	- Take in large amount of input data, runs job, and produces output
	- Metrics
		- Throughput (primary)
- Stream processing Systems (near real-time)
	- Somewhere betwee online / batch processing
	- Opeartes on events shortly after they happen




