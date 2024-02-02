---
title: How React Handles State
---
# Render and Commit
Before components are displayed on the screen, they must be rendered by React

Requesting / serving UI has 3 steps
1. Triggering render (delivering order to kitchen)
2. Rendering component (preparing order in the kitchen)
3. Comitting to the DOM (placing order on the table)


**1. Triggering a render**
Happens for two reasons
1. Initial render: when app starts
2. Re-render: When state updates
	1. Updating state automatically queues a render

**2. React Renders your components**
React figures out what to display on the screen by calling your components

On initial render- react calls root component
Subsequent render- calls function component whose state update triggered render
- If has children, recursive rendering of component’s children

**3. React Commits changes to the DOM**
After rendering, the actual modifying of the DOM

Initial render- uses `appendChild()` DOM API to put all DOM nodes it has on screen

Re-renders: minmial necessary operations (calculated while rendering) to make DOM match the latest rendering output
- Only changes DOM nodes if difference between renders


# State as a Snapshot
Think of state as a snapshot
- Updating state doesn’t change the state varibale you have, but triggers a rerender
- Thus, if you change state then immediately log, the state will seem the same


Remember: Setting state triggers a re-render

Rendering takes a snapshot in time, when React re-renders a component
1. React calls your function again
2. Your function returns a new JSX snapshot
3. React then updates the screen to match the snapshot your function returned

State lives in React itself

Setting state only changes it for the next render (thus, doing `setNumber(number+1)` 10 times is the same as once)

Event handlers created in the past have state values from the render inw hich they were created
