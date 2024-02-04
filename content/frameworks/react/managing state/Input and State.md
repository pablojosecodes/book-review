---
title: Input and State
---
React fundamentally changes how we build user interfaces by shifting from imperative to declarative programming. This shift is crucial for managing complex UIs. Let's explore how React enables this.
# Reacting to Input with State
Imperatively manipulating UI gets very complicated
- This is exactly what React solves
In React- you just (declarative) declare what you want to show and React figures out how to update the UI

Thinking about UI declaratively
1. Identify component’s visual states (e.g. typing, empty, error, success, etc.)
2. Determine what triggers those state changes (inputs from humans + computers)
	2. Often use event handlers for huma inputs
	3. Can draw a flow with arrows between states (ie. “Start typing”, “press submit”, etc.)
3. Represent the state  with `useState` + refactor
	1. Simple as possible
	2. Start with state that absolutely must be there
	3. Make sure you’re only tracking what is essential (ie. combine typing, submitting, success into a single status state)
Note: to model state more precisely- can extract it into a reducer
4. Connect event handlers to set state

