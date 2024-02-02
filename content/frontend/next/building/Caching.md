---
title: Caching
---
Caching is essential for imporving performance and reducing costs. You can use NextJS to cache rendering work and data requests. Let’s check out how.

> Note: most caching is done by default in NextJS, so this section is more for understanding the internals of NextJS 

# Quick overview

NextJS caches as much as it can
- Routes: statically rendereed
- Data requests: default cached

Caching mechanisms in NextjS $\Downarrow$ 

| Mechanism           | Caches                         | Where  | Why?                                           | Longevity                         |
|---------------------|------------------------------|--------|---------------------------------------------------|----------------------------------|
| Request Memoization | Return values of functions   | Server | Re-use data in React Component tree             | Per-request lifecycle            |
| Data Cache          | Data                         | Server | Store data across user requests / deployments   | Persistent (can be revalidated)  |
| Full Route Cache    | HTML and RSC payload         | Server | Reduce rendering cost and improve performance     | Persistent (can be revalidated)  |
| Router Cache        | RSC Payload                  | Client | Reduce server requests on navigation              | User session or time-based       |
# Request Memoization
# Data Cache
# Full Route Cache
# Router Cache
# Cache Interactions

