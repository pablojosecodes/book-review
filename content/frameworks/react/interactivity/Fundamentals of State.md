---
title: Fundamentals of State
---

# State

Use `useState` to remember things
- Is a hook- special function which lets component use React features

Naive state: Just changing variables- doesn’t work
- Local varaibles don’t persist between renders
- Changes to local variables won’t trigger renders
`useState` fixes both
- State variable: retains data between data with a state variable
- State setter function: updates variable and triggers React to render component again

Convention
- `const [property, setProperty] = useState(initialProperty)`

Rule: only call hooks at the top level
- React relies on stable call order on every render of the same component to know which state is which

State is private + independent for each component

