import axios from 'axios'

const http = axios.create({ baseURL: '/api' })

export const getTickers = () => http.get('/tickers').then(r => r.data)
export const getCEO    = (ticker) => http.get(`/ceo/${ticker}`).then(r => r.data)
export const getCalls  = (ticker) => http.get(`/calls/${ticker}`).then(r => r.data)
export const getLatestCall = (ticker) => http.get(`/calls/${ticker}/latest`).then(r => r.data)
export const getFlags  = () => http.get('/flags').then(r => r.data)
export const scoreCall = (ticker, call_date) =>
  http.post(`/calls/${ticker}/score`, { call_date }).then(r => r.data)
export const getHealth = () => http.get('/health').then(r => r.data)

export const CEO_NAMES = {
  AAPL: 'Tim Cook',
  LLY:  'David Ricks',
  PFE:  'Albert Bourla',
  SHOP: 'Tobias Lütke',
  GM:   'Mary Barra',
  SBUX: 'Brian Niccol',
  DUOL: 'Luis von Ahn',
}
