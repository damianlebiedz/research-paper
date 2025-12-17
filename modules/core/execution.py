from modules.core.models import Pair, PositionState, StrategyParams


class TradeExecutor:
    @staticmethod
    def get_spread(x: str, y: str, position: float) -> tuple[float, float]:  # TODO
        """Get spread for two assets depending on position."""
        if position == 0:
            # SPREAD FOR POSITION CLOSING
            return 1.0, 1.0
        elif position > 0:
            # SPREAD FOR POSITIVE POSITION OPENING
            return 1.0, 1.0
        else:
            # SPREAD FOR NEGATIVE POSITION OPENING
            return 1.0, 1.0

    @classmethod
    def execute(cls, pair: Pair, position_state: PositionState, strategy_params: StrategyParams, price_x: float,
                price_y: float, z_score: float, beta: float, total_fees: float) -> tuple[float, float]:
        """
        Main execution method.
        Returns: (pnl_from_this_step, updated_total_fees)
        Modifies the position_state object in place.
        """

        # IN POSITION
        if position_state.prev_position != 0:
            # CLOSE POSITION (STOP LOSS OR TAKE PROFIT)
            if (
                    position_state.prev_position < 0 and (
                    z_score <= strategy_params.exit_threshold or (
                    position_state.stop_loss_threshold is not None and z_score >= position_state.stop_loss_threshold))) or (
                    position_state.prev_position > 0 and (
                    z_score >= -strategy_params.exit_threshold or (
                    position_state.stop_loss_threshold is not None and z_score <= -position_state.stop_loss_threshold))
            ):
                return cls._close_position(pair, position_state, price_x, price_y, total_fees)

            # HOLD POSITION
            else:
                return cls._hold_position(position_state, price_x, price_y, total_fees)

        # OUT OF POSITION
        else:
            # OPEN POSITION
            if position_state.position != 0:
                return cls._open_position(
                    beta, z_score, pair, position_state, strategy_params, price_x, price_y, total_fees
                )
            # STAY OUT OF POSITION
            else:
                return 0, total_fees

    @classmethod
    def _open_position(cls, beta, z_score, pair, position_state, strategy_params, price_x, price_y,
                       total_fees) -> tuple[float, float]:
        wx = 1 / (beta + 1)
        wy = 1 - wx

        position_cash = abs(position_state.position) * pair.initial_cash
        x_spread, y_spread = cls.get_spread(pair.x, pair.y, position_state.position)

        if position_state.position > 0:
            qx = position_cash * wx / (price_x * x_spread)
            qy = -(position_cash * wy) / (price_y * y_spread)
        elif position_state.position < 0:
            qx = -(position_cash * wx) / (price_x * x_spread)
            qy = position_cash * wy / (price_y * y_spread)
        else:
            raise ValueError("Position cannot be 0 while opening")

        entry_value = abs(qx) * price_x + abs(qy) * price_y
        if strategy_params.stop_loss is not None:
            stop_loss_thr = abs(z_score * strategy_params.stop_loss)
        else:
            stop_loss_thr = None

        position_state.update_position(position_state.position, position_state.prev_position, qx,
                                       qy, wx, wy, entry_value, stop_loss_thr)

        pos_fees = entry_value * pair.fee_rate
        t_fees = total_fees + pos_fees
        return 0, t_fees

    @classmethod
    def _close_position(cls, pair, position_state, price_x, price_y, total_fees) -> tuple[float, float, dict]:
        x_spread, y_spread = cls.get_spread(pair.x, pair.y, 0)

        exit_value = abs(position_state.q_x) * (price_x * x_spread) + abs(position_state.q_y) * (price_y * y_spread)
        pos_fees = exit_value * pair.fee_rate

        if position_state.prev_position > 0:
            pnl = exit_value - position_state.entry_val
        elif position_state.prev_position < 0:
            pnl = position_state.entry_val - exit_value
        else:
            raise ValueError("Position cannot be 0 while closing")

        position_state.clear_position()

        t_fees = total_fees + pos_fees
        return pnl, t_fees

    @staticmethod
    def _hold_position(position_state, price_x, price_y, total_fees):
        curr_val = abs(position_state.q_x) * price_x + abs(position_state.q_y) * price_y

        if position_state.prev_position > 0:
            pnl = curr_val - position_state.entry_val
        else:
            pnl = position_state.entry_val - curr_val

        position_state.position = position_state.prev_position

        return pnl, total_fees
