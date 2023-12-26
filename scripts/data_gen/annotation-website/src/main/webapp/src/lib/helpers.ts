export function projectStatusColor(status: any) {
  switch (status) {
    case "PREVIEW":
      return "warning";
    case "OPEN":
      return "success";
    case "CLOSED":
      return "danger";
    default:
      return "secondary";
  }
}

export function getTeamColor(team: string | undefined) {
  return team === "Blue" ? 'lightblue' :
    team === "Red" ? 'lightpink' : undefined
}