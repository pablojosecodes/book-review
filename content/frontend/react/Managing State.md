---
title: Managing State
---

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

# Sharing State between Components
Sometimes you want state between 2 components to always change togethrer
- Life state up to the closest parent

3 steps for this
1. Remove state from the child components
	1. Remove `useState` and pass state as prop
2. Pass hardcoded data from the common parent to chidren props
3. Add state to common parent

Uncontrolled + controlled components
- “uncontrolled”: Component with some local state
- “controlled”: when important info is driven by props

Single source of truth for each state
- For each unique piece of state- choose the component that “owns” it

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







# Extracting State Login into a Reducer
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

# Passing Data Deeply with Context
Context as an alternative to passing props
- make info availabel to any component below without explicit props usage

Issue: verbose if need to pass prop through many components or many comonents need the same prop

Problem with props: becomes verbose when need to pass far down

Example context usage
- You have a bunch of  `<Heading>` components in a `<Section>` component and want each to now the level
Process

**1. Create the context**
Create it using `createContext` and export it so componetnts can use it
Sample → 
```typescript
import { createContext } from 'react';

export const LevelContext = createContext(1);
```

Only argument is the default value

**2. Use the context**
Import the `useContext` hook and the context
Then, read it `const level = useContext(LevelContext)`

**3. Provide the context**
Wrap with a context provider
```typescript
  <LevelContext.Provider value={level}>
	{children}
  </LevelContext.Provider>
```
This says
- If component asks for `levelContext`, give them this level

**You can use/provide context from the same component**
You can have each sectino read the `level` from the section above and pass `level+1` down automatically.
Basically same as above, but then `<LevelContext.Provider value={level+1}>`

**Context passes through intermediate components**
Have as many intermediate components as you want!

**Before using context**
1. Start with props- will make clear which components use which data
2. Extract comonents and pass JSX as children to them
If those don’t work, use context

**Good use cases for context**
- Theming
- Current account
- Routing
- Managing state in complex applications

# Scaling Up with Reducer and Context
Combine reducers and context to manage complex state!


**Step 1: create the context**
Define reducer
- `const [tasks, dispatch] = useReducer(tasksReducer, initialTasks);`
Create 2 contexts: export from a separate file
- `TasksContext`: current list of tasks
	- e.g. `export const TasksContext = createContext(null);`
- `TasksDispatchContext`: function which lets components dispatch actions
	- e.g. `export const TasksDispatchContext = createContext(null);`

**Step 2: Put state and dispatch into context**
Take tasks and dispatch returned by `useReducer` and provide to entire tree
```typescript
  const [tasks, dispatch] = useReducer(tasksReducer, initialTasks);
  // ...
  return (
    <TasksContext.Provider value={tasks}>
      <TasksDispatchContext.Provider value={dispatch}>
        ...
      </TasksDispatchContext.Provider>
    </TasksContext.Provider>
  );
```

**Step 3: use context anywhere in the tree**

e.g.
- Read task list from the `TaskContext`
- Update task list from the `TaskDispatchContext`


**Organizational stuff**
Declutter by moving reducer and context into a single file
- Can create a Provider which takes in children

Sample → 
```typescript
export function TasksProvider({ children }) {
  const [tasks, dispatch] = useReducer(tasksReducer, initialTasks);

  return (
    <TasksContext.Provider value={tasks}>
      <TasksDispatchContext.Provider value={dispatch}>
        {children}
      </TasksDispatchContext.Provider>
    </TasksContext.Provider>
  );
}
```
