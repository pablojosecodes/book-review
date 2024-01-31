---
title: Structuring State with components
---
# Choosing the State Structure
State structure//shape can be hugely important

Some principles
1. Group related state
2. Avoid contradictions- don’t leave room for mistakes
3. Avoid redundant state
4. Avoid duplication in state
5. Avoid deeply nested state


**Group related state**
Issue: forgetting to keep multiple variables in sync

If 2+ state variables always change together, consider unifying them into a single state variable
Example: x-y coordinates
Another example: when don’t know how many pieces of state you’ll need- ie. a form

**Avoid contradictions**
Issue: leaving room for mistakes

Just don’t make it possible for impossible state combinations to occur
- For example- replace isSent and isSending with a singular sendingStatus 

 **Avoid redundant state**
 If you can calculate info from existing state variables during rendering, don’t put that info into component’s stae
 - e.g. firstname + lastname (don’t use state for fullname!)
 - instead, make `const fullName = firstName + ' ' + lastname;` 
	 - This will update during render

**Don’t mirror props in state**
- Don’t define state using prop (state is only initialized during render)
	- Unless, you specifically want to ignore all updates for a specific prop
- Instead, use a constant


**Avoid duplication in state**
Make it so you never need to store same state multiple places

**Avoid deeply nested state**
Keep as flat as possible

How to store object with Planet → Continent → Country? How to update?
If facing issue like this (to much nesting for easy updating), consider making it flat
- Store ids instead of objects 
- Now only need to update 2 levels of state
	- Updated version of parent
	- Updated version of root able object
- Example of this
```typescript

  function handleComplete(parentId, childId) {
    const parent = plan[parentId];
    // Create a new version of the parent place
    // that doesn't include this child ID.
    const nextParent = {
      ...parent,
      childIds: parent.childIds
        .filter(id => id !== childId)
    };
    // Update the root state object...
    setPlan({
      ...plan,
      // ...so that it has the updated parent.
      [parentId]: nextParent
    });
```

Can also use Immer here
# Preserving and Resetting State
When you re-render, React decides which parts of render tree to 
- Keep + Update
- Or: Discard + Recreate

Not always perfect (e.g. switching between texting different people)

You can force component ro reset state
- Pass a different key- treats as a different component to be re-created from scratch

**State is tied to position in render tree**
React builds render trees for the component structure in your UI
Given a component state
- State is held in React- associates each piece of state with correct componetnt by where component sits in the render tree
- Each component has fully isolated state
When you stop rendering / remove a component, its state is destroyed
- Also state of subtree


But the same component at the same position preseres state (e.g. same key but differnt styles)
- POSITION 
- But also component types

Say you want to reset state when switching between components
1. Render component in different positions
```typescript
{isPlayerA &&
	<Counter person="Taylor" />
}
{!isPlayerA &&
	<Counter person="Sarah" />
}
```

2. Reset state with a key
```typescript
  {isPlayerA ? (
	<Counter key="Taylor" person="Taylor" />
  ) : (
	<Counter key="Sarah" person="Sarah" />
  )}
```



To preserve state for removed components
Couple of methods
1. Render all chats but hide othrs
2. Life state up
3. Use a different source (e.g. `localStorage`)







# Extracting State Logic into a Reducer
Many state updates across event handlers gets overwhelming
Solution: consolidate update logic outside of component into a single function- a **reducer**

**Consolidate state logic with a reducer**
Example use case
- Array of tasks
- Event handlers for adding, for removing, and for editing
	- Each call `setTasks`
3 steps to migrate from `useState` → `reducer`
1. Setting state → dispatching actions
	1. Prev: handlers specify what to do by setting the state
		1. `handleChangeTask(task){ code to change task}`
	2. Now: specify what the user did by dispatching an **action**
		1. `handleChangeTask(task){ dispatch({type: 'changed', task: task,})};`
		2. Note: action can have any shape
2. Write a reducer function `function yourReducer(state, action){ return next state };`
	1. Takes in current state and action as arguments
	2. Returns next state
3. Use the reducer from your component
	1. `import { useReducer } from 'react';`
	2. `const [tasks, dipatch] = useReducer(tasksReducer, initialTasks);`
	3. `useReducer` 
		1. Arguments: reducer function, inital state
		2. Returns: stateful value, dispatch function
4. 

Since reducer function takes the state as an argument- you can declare it outside of your component!


**Notes on useState vs useReducer**
- Code size:
	- `useReducer` has more code up front- need to write reducer function + dispatch actions
- Readbility
	- `useReducer` helps with complex state updates
- Debugging: can add console logs into reducer
- Testing: You can export + test reducer in isolution

**Writing reducers well**
1. Must be pure! They run during rendering
2. Each action describes a single user interaction (even if multiple changes occur in data)
3. Use Immer to write concise reducers
