---
title: Defining Events
---

# Responding to events

Event handlers: functions triggered in response to user interactions
- Built-in components’s like `<button>` only support built-in browser-event slike `onclick`
- You can build your own event handlers
- Name typically begins with “handle”


Adding event handler
- Pass as prop
- Can also define inline with arrow function `onclick={() => {function();}}`

Pitfall: Functions passed to event handlers must be passed, not called
- ie `<button onClick={handleClick}>`

Props
- You can pass prop as the event handler

Naming event handler props
- Convention: start with on

Note: make sure to use appropriate HTML tags for event handlers

Event propagation
- Events bubble up the tree- one by one
- Except for `onScroll`- only works on JSX tag its attached to

Stopping propagation
- You can call `e.stopPropagation();` to stop event propagation from going further
- Note: if yu really need to catch all events on child elements, even if they stopped propagation (ie. for logging every click for analytics),  you can use `Capture` at end of the event name (ie. `onClickCapture`)
	- Thus, each event propagates in 3 phases
		- Travels downwards, calling all `onClickCapture` handleres
		- Runs clicked element’s `onClick` handler
		- Travels up, calling all `onClick` handlers


Preventing default behavior
- Use `e.preventDefault()` to prevent browser default behavior

