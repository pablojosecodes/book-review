---
title: Communicating State
---
State doesn’t always live exclusively within a single component and it can be useful (even essential) to communicate a shared state between components or across an entire React project. Let’s explore how to do this effectively.
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
